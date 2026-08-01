#!/usr/bin/env python3
"""Official ABSA-QUAD Rest15/Rest16 reader and surface-span alignment.

The released benchmark stores surface-form quadruples as::

    sentence####[[aspect, category, polarity, opinion], ...]

``NULL`` denotes an implicit element.  This module maps explicit surfaces to
half-open character spans with an auditable, unique case-insensitive fallback.
Repeated surfaces are resolved deterministically by ranking every
aspect/opinion occurrence pair by strong-clause barriers, character gap,
centre distance, and then stable left-to-right offsets.  Overlap is valid: an
opinion may be contained in its aspect surface.  Only when the two released
surface strings are identical and distinct role spans exist are same-span
pairs removed.  Duplicate raw quadruples consume the next ranked occurrence
pair when one is available.  Every ambiguity and unrepresentable annotation
remains auditable.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence


REST_DATASETS = ("rest15", "rest16")
REST_SPLITS = ("train", "dev", "test")
REST_CATEGORY_VOCAB = (
    "location general",
    "food prices",
    "food quality",
    "food general",
    "ambience general",
    "service general",
    "restaurant prices",
    "drinks prices",
    "restaurant miscellaneous",
    "drinks quality",
    "drinks style_options",
    "restaurant general",
    "food style_options",
)
REST_CATEGORY_TO_ID = {
    category: index for index, category in enumerate(REST_CATEGORY_VOCAB)
}
REST_SENTIMENT_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
REST_ID_TO_SENTIMENT = {
    value: key for key, value in REST_SENTIMENT_TO_ID.items()
}
NUM_REST_CATEGORIES = len(REST_CATEGORY_VOCAB)
DEFAULT_REST_MAX_LENGTH = 128
REST_PAIR_LABEL_TO_ID = {"NONE": 0, "negative": 1, "neutral": 2, "positive": 3}
REST_PAIR_ID_TO_SENTIMENT = {
    value: key for key, value in REST_PAIR_LABEL_TO_ID.items() if key != "NONE"
}
_STRONG_BARRIER_RE = re.compile(r"--|[;:.!?()\[\]{}\u2013\u2014]")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REST_QUAD_ROOT = (
    PROJECT_ROOT / "data" / "asqp"
)
EXPECTED_SENTENCE_COUNTS = {
    "rest15": {"train": 834, "dev": 209, "test": 537},
    "rest16": {"train": 1264, "dev": 316, "test": 544},
}


@dataclass(frozen=True)
class RestCharSpan:
    start: int | None
    end: int | None

    def __post_init__(self) -> None:
        if (self.start is None) != (self.end is None):
            raise ValueError("A RestCharSpan must have two offsets or be NULL")
        if self.start is not None and (self.start < 0 or self.end <= self.start):
            raise ValueError(f"Invalid character span: {self.start},{self.end}")

    @classmethod
    def null(cls) -> "RestCharSpan":
        return cls(None, None)

    @property
    def is_null(self) -> bool:
        return self.start is None

    def as_tuple(self) -> tuple[int, int] | None:
        if self.is_null:
            return None
        assert self.start is not None and self.end is not None
        return self.start, self.end

    def surface(self, text: str) -> str:
        if self.is_null:
            return "NULL"
        assert self.start is not None and self.end is not None
        return text[self.start : self.end]


@dataclass(frozen=True)
class RestSurfaceQuad:
    aspect: str
    category: str
    sentiment: str
    opinion: str

    def __post_init__(self) -> None:
        if not self.aspect or not self.opinion:
            raise ValueError("Aspect/opinion surfaces cannot be empty")
        if self.opinion == "NULL":
            raise ValueError(
                "ABSA-QUAD Rest15/Rest16 exclude implicit opinions"
            )
        if self.category not in REST_CATEGORY_TO_ID:
            raise ValueError(f"Unknown Rest category: {self.category!r}")
        if self.sentiment not in REST_SENTIMENT_TO_ID:
            raise ValueError(f"Unknown Rest sentiment: {self.sentiment!r}")

    def as_tuple(self) -> tuple[str, str, str, str]:
        return self.aspect, self.category, self.sentiment, self.opinion


@dataclass(frozen=True)
class RestQuadAnnotation:
    raw: RestSurfaceQuad
    aspect_span: RestCharSpan | None
    opinion_span: RestCharSpan | None
    alignment_status: str
    alignment_reason: str | None
    aspect_occurrences: int
    opinion_occurrences: int
    aspect_match_method: str
    opinion_match_method: str
    candidate_pair_count: int
    selected_candidate_rank: int | None
    duplicate_raw_rank: int
    reused_candidate_pair: bool = False

    @property
    def is_representable(self) -> bool:
        return (
            self.alignment_status == "aligned"
            and self.aspect_span is not None
            and self.opinion_span is not None
        )

    def span_key(
        self,
    ) -> tuple[tuple[int, int] | None, str, str, tuple[int, int] | None]:
        if not self.is_representable:
            raise ValueError("Unrepresentable annotation has no span key")
        assert self.aspect_span is not None and self.opinion_span is not None
        return (
            self.aspect_span.as_tuple(),
            self.raw.category,
            self.raw.sentiment,
            self.opinion_span.as_tuple(),
        )


@dataclass(frozen=True)
class RestQuadExample:
    text: str
    annotations: tuple[RestQuadAnnotation, ...]
    source_path: str | None = None
    line_no: int | None = None
    boundary_whitespace_normalizations: int = 0

    @property
    def raw_quads(self) -> tuple[RestSurfaceQuad, ...]:
        return tuple(annotation.raw for annotation in self.annotations)

    @property
    def representable_annotations(self) -> tuple[RestQuadAnnotation, ...]:
        return tuple(
            annotation
            for annotation in self.annotations
            if annotation.is_representable
        )


@dataclass
class RestQuadReadStats:
    lines_seen: int = 0
    examples_parsed: int = 0
    quads_seen: int = 0
    representable_quads: int = 0
    unrepresentable_quads: int = 0
    null_aspects: int = 0
    null_opinions: int = 0
    ambiguous_aspect_quads: int = 0
    ambiguous_opinion_quads: int = 0
    ambiguous_pair_quads: int = 0
    duplicate_raw_quads: int = 0
    duplicate_pair_disambiguations: int = 0
    reused_candidate_pairs: int = 0
    case_insensitive_aspect_fallbacks: int = 0
    case_insensitive_opinion_fallbacks: int = 0
    missing_aspect_surfaces: int = 0
    missing_opinion_surfaces: int = 0
    invalid_lines: int = 0
    boundary_whitespace_normalizations: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            field_name: int(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class RestQuadIssue:
    source_path: str
    line_no: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.source_path} line {self.line_no}: [{self.code}] {self.message}"


@dataclass(frozen=True)
class RestQuadReadResult(Sequence[RestQuadExample]):
    examples: tuple[RestQuadExample, ...]
    stats: RestQuadReadStats
    issues: tuple[RestQuadIssue, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def __iter__(self) -> Iterator[RestQuadExample]:
        return iter(self.examples)


class RestQuadFormatError(ValueError):
    pass


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def find_surface_occurrences(text: str, surface: str) -> tuple[RestCharSpan, ...]:
    """Return exact, case-sensitive, token-boundary-compatible occurrences."""

    if surface == "NULL":
        return (RestCharSpan.null(),)
    occurrences = []
    search_from = 0
    while True:
        start = text.find(surface, search_from)
        if start < 0:
            break
        end = start + len(surface)
        left_ok = (
            start == 0
            or not _is_word_character(surface[0])
            or not _is_word_character(text[start - 1])
        )
        right_ok = (
            end == len(text)
            or not _is_word_character(surface[-1])
            or not _is_word_character(text[end])
        )
        if left_ok and right_ok:
            occurrences.append(RestCharSpan(start, end))
        search_from = start + 1
    return tuple(occurrences)


def _case_insensitive_occurrences(
    text: str, surface: str
) -> tuple[RestCharSpan, ...]:
    occurrences = []
    for match in re.finditer(re.escape(surface), text, flags=re.IGNORECASE):
        start, end = match.span()
        left_ok = (
            start == 0
            or not _is_word_character(surface[0])
            or not _is_word_character(text[start - 1])
        )
        right_ok = (
            end == len(text)
            or not _is_word_character(surface[-1])
            or not _is_word_character(text[end])
        )
        if left_ok and right_ok:
            occurrences.append(RestCharSpan(start, end))
    return tuple(occurrences)


def _surface_candidates(
    text: str, surface: str
) -> tuple[tuple[RestCharSpan, ...], str]:
    exact = find_surface_occurrences(text, surface)
    if exact:
        return exact, "exact"
    if surface == "NULL":
        return exact, "null"
    folded = _case_insensitive_occurrences(text, surface)
    if len(folded) == 1:
        return folded, "unique_case_insensitive_fallback"
    if folded:
        return (), "ambiguous_case_insensitive_rejected"
    return (), "missing"


def _pair_rank(
    text: str,
    aspect: RestCharSpan,
    opinion: RestCharSpan,
) -> tuple[int, int, float, int, int, int]:
    if aspect.is_null and opinion.is_null:
        return 0, 0, 0.0, -1, -1, -1
    if aspect.is_null:
        assert opinion.start is not None
        return 0, 0, 0.0, opinion.start, opinion.start, -1
    if opinion.is_null:
        assert aspect.start is not None
        return 0, 0, 0.0, aspect.start, -1, aspect.start
    assert aspect.start is not None and aspect.end is not None
    assert opinion.start is not None and opinion.end is not None
    gap = max(opinion.start - aspect.end, aspect.start - opinion.end, 0)
    if aspect.end <= opinion.start:
        between = text[aspect.end : opinion.start]
    elif opinion.end <= aspect.start:
        between = text[opinion.end : aspect.start]
    else:
        between = ""
    barriers = len(_STRONG_BARRIER_RE.findall(between))
    aspect_centre = (aspect.start + aspect.end) / 2.0
    opinion_centre = (opinion.start + opinion.end) / 2.0
    return (
        barriers,
        int(gap),
        abs(aspect_centre - opinion_centre),
        min(aspect.start, opinion.start),
        opinion.start,
        aspect.start,
    )


def align_surface_quads(
    text: str,
    raw_quads: Sequence[RestSurfaceQuad],
) -> tuple[RestQuadAnnotation, ...]:
    """Align one sentence using the frozen deterministic disambiguation rule."""

    duplicate_rank: Counter[tuple[str, str, str, str]] = Counter()
    used_pairs: dict[
        tuple[str, str, str, str],
        set[tuple[RestCharSpan, RestCharSpan]],
    ] = {}
    used_aspects: dict[tuple[str, str, str, str], set[RestCharSpan]] = {}
    used_opinions: dict[tuple[str, str, str, str], set[RestCharSpan]] = {}
    annotations = []
    for raw in raw_quads:
        raw_key = raw.as_tuple()
        occurrence_rank = duplicate_rank[raw_key]
        duplicate_rank[raw_key] += 1
        aspect_occurrences, aspect_method = _surface_candidates(text, raw.aspect)
        opinion_occurrences, opinion_method = _surface_candidates(text, raw.opinion)
        missing = []
        if not aspect_occurrences:
            missing.append("aspect")
        if not opinion_occurrences:
            missing.append("opinion")
        if missing:
            annotations.append(
                RestQuadAnnotation(
                    raw=raw,
                    aspect_span=None,
                    opinion_span=None,
                    alignment_status="unrepresentable",
                    alignment_reason="missing_" + "_and_".join(missing) + "_surface",
                    aspect_occurrences=len(aspect_occurrences),
                    opinion_occurrences=len(opinion_occurrences),
                    aspect_match_method=aspect_method,
                    opinion_match_method=opinion_method,
                    candidate_pair_count=0,
                    selected_candidate_rank=None,
                    duplicate_raw_rank=occurrence_rank,
                )
            )
            continue
        candidate_pairs = [
            (aspect, opinion)
            for aspect in aspect_occurrences
            for opinion in opinion_occurrences
        ]
        if raw.aspect == raw.opinion:
            distinct_role_pairs = [
                pair for pair in candidate_pairs if pair[0] != pair[1]
            ]
            if distinct_role_pairs:
                candidate_pairs = distinct_role_pairs
        candidates = sorted(
            candidate_pairs,
            key=lambda pair: _pair_rank(text, *pair),
        )
        selected_rank = 0
        if occurrence_rank > 0:
            prior_aspects = used_aspects.get(raw_key, set())
            prior_opinions = used_opinions.get(raw_key, set())
            unused_anchor_ranks = [
                index
                for index, (aspect, opinion) in enumerate(candidates)
                if (aspect.is_null or aspect not in prior_aspects)
                and (opinion.is_null or opinion not in prior_opinions)
            ]
            unused_pair_ranks = [
                index
                for index, pair in enumerate(candidates)
                if pair not in used_pairs.get(raw_key, set())
            ]
            if unused_anchor_ranks:
                selected_rank = unused_anchor_ranks[0]
            elif unused_pair_ranks:
                selected_rank = unused_pair_ranks[0]
        aspect_span, opinion_span = candidates[selected_rank]
        used_pairs.setdefault(raw_key, set()).add((aspect_span, opinion_span))
        used_aspects.setdefault(raw_key, set()).add(aspect_span)
        used_opinions.setdefault(raw_key, set()).add(opinion_span)
        annotations.append(
            RestQuadAnnotation(
                raw=raw,
                aspect_span=aspect_span,
                opinion_span=opinion_span,
                alignment_status="aligned",
                alignment_reason=None,
                aspect_occurrences=len(aspect_occurrences),
                opinion_occurrences=len(opinion_occurrences),
                aspect_match_method=aspect_method,
                opinion_match_method=opinion_method,
                candidate_pair_count=len(candidates),
                selected_candidate_rank=selected_rank,
                duplicate_raw_rank=occurrence_rank,
                reused_candidate_pair=(
                    occurrence_rank > 0
                    and sum(
                        1
                        for previous in annotations
                        if previous.raw.as_tuple() == raw_key
                        and previous.aspect_span == aspect_span
                        and previous.opinion_span == opinion_span
                    )
                    > 0
                ),
            )
        )
    return tuple(annotations)


def parse_rest_quad_line(
    line: str,
    *,
    source_path: str | Path | None = None,
    line_no: int | None = None,
) -> RestQuadExample:
    raw_line = line.rstrip("\r\n")
    if not raw_line.strip() or "####" not in raw_line:
        raise RestQuadFormatError("Line must contain sentence####quadruples")
    text, literal = raw_line.split("####", 1)
    if not text:
        raise RestQuadFormatError("Sentence text is empty")
    try:
        payload = ast.literal_eval(literal)
    except (SyntaxError, ValueError) as exc:
        raise RestQuadFormatError("Invalid quadruple literal") from exc
    if not isinstance(payload, (list, tuple)):
        raise RestQuadFormatError("Quadruple payload must be a list")
    raw_quads = []
    boundary_whitespace_normalizations = 0
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise RestQuadFormatError(f"Malformed quadruple: {item!r}")
        raw_values = tuple(str(value) for value in item)
        values = tuple(value.strip() for value in raw_values)
        boundary_whitespace_normalizations += sum(
            int(raw != normalized)
            for raw, normalized in zip(raw_values, values)
        )
        aspect, category, sentiment, opinion = values
        try:
            raw_quads.append(
                RestSurfaceQuad(
                    aspect=aspect,
                    category=category,
                    sentiment=sentiment,
                    opinion=opinion,
                )
            )
        except ValueError as exc:
            raise RestQuadFormatError(str(exc)) from exc
    annotations = align_surface_quads(text, raw_quads)
    return RestQuadExample(
        text=text,
        annotations=annotations,
        source_path=str(source_path) if source_path is not None else None,
        line_no=line_no,
        boundary_whitespace_normalizations=boundary_whitespace_normalizations,
    )


def rest_split_path(
    dataset: str,
    split: str,
    *,
    data_root: str | Path = DEFAULT_REST_QUAD_ROOT,
) -> Path:
    if dataset not in REST_DATASETS:
        raise ValueError(f"dataset must be one of {REST_DATASETS}")
    if split not in REST_SPLITS:
        raise ValueError(f"split must be one of {REST_SPLITS}")
    return Path(data_root) / dataset / f"{split}.txt"


def read_rest_quad_split(
    dataset: str,
    split: str,
    *,
    data_root: str | Path = DEFAULT_REST_QUAD_ROOT,
    strict_format: bool = True,
) -> RestQuadReadResult:
    path = rest_split_path(dataset, split, data_root=data_root)
    stats = RestQuadReadStats()
    examples = []
    issues = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stats.lines_seen += 1
            try:
                example = parse_rest_quad_line(
                    line,
                    source_path=path,
                    line_no=line_no,
                )
            except RestQuadFormatError as exc:
                stats.invalid_lines += 1
                issues.append(
                    RestQuadIssue(str(path), line_no, "invalid_format", str(exc))
                )
                continue
            stats.examples_parsed += 1
            stats.boundary_whitespace_normalizations += (
                example.boundary_whitespace_normalizations
            )
            stats.quads_seen += len(example.annotations)
            raw_counts = Counter(annotation.raw.as_tuple() for annotation in example.annotations)
            stats.duplicate_raw_quads += sum(
                max(0, count - 1) for count in raw_counts.values()
            )
            for annotation in example.annotations:
                stats.null_aspects += int(annotation.raw.aspect == "NULL")
                stats.null_opinions += int(annotation.raw.opinion == "NULL")
                stats.ambiguous_aspect_quads += int(annotation.aspect_occurrences > 1)
                stats.ambiguous_opinion_quads += int(annotation.opinion_occurrences > 1)
                stats.ambiguous_pair_quads += int(annotation.candidate_pair_count > 1)
                stats.duplicate_pair_disambiguations += int(
                    annotation.duplicate_raw_rank > 0 and annotation.is_representable
                )
                stats.reused_candidate_pairs += int(annotation.reused_candidate_pair)
                stats.case_insensitive_aspect_fallbacks += int(
                    annotation.aspect_match_method
                    == "unique_case_insensitive_fallback"
                )
                stats.case_insensitive_opinion_fallbacks += int(
                    annotation.opinion_match_method
                    == "unique_case_insensitive_fallback"
                )
                if annotation.is_representable:
                    stats.representable_quads += 1
                else:
                    stats.unrepresentable_quads += 1
                    reason = annotation.alignment_reason or "unknown"
                    stats.missing_aspect_surfaces += int("aspect" in reason)
                    stats.missing_opinion_surfaces += int("opinion" in reason)
                    issues.append(
                        RestQuadIssue(
                            str(path),
                            line_no,
                            "unrepresentable_surface",
                            f"{annotation.raw.as_tuple()!r}: {reason}",
                        )
                    )
            examples.append(example)
    if strict_format and stats.invalid_lines:
        preview = "; ".join(str(issue) for issue in issues[:3])
        raise RestQuadFormatError(
            f"{stats.invalid_lines} invalid lines in {path}: {preview}"
        )
    expected = EXPECTED_SENTENCE_COUNTS[dataset][split]
    if len(examples) != expected:
        raise RestQuadFormatError(
            f"Unexpected {dataset}/{split} size: {len(examples)} != {expected}"
        )
    return RestQuadReadResult(tuple(examples), stats, tuple(issues))
