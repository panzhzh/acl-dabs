#!/usr/bin/env python3
# scripts/convert_sem16_other_datasets.py
# -*- coding: utf-8 -*-

"""
Convert SemEval-2016 Dutch/Turkish datasets under `data/raw-sem16-other/` into
this repo's JSON format under `data/semeval/`, following existing naming.

Datasets covered:
- Restaurant_16_DU: term-level Dutch restaurant (explicit targets/spans)
- Restaurant_16_TU: term-level Turkish restaurant (explicit targets/spans)
- Phone_16_DU: category-level Dutch phone/smartphone (no explicit targets/spans)
  For Phone_16_DU we export aspect categories as `term` tokens and set `from/to`
  to null.

Outputs:
- `data/semeval/Restaurant_16_{DU,TU}/Restaurants_{Train,Test}.csv`
- `data/semeval/Restaurant_16_{DU,TU}/Restaurants_{Train,Test}.json`
- `data/semeval/Phone_16_DU/Phones_{Train,Test}.json`
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Add repo root so `import src...` works when running `python scripts/...py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_converter import get_stanza_pipeline
from src.utils.data_converter import convert_csv_to_json
from src.utils.res16_to_semeval_converter import convert_xml_to_csv


def _category_to_tokens(category: str) -> list[str]:
    # Example: "MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE" -> ["multimedia", "devices", "operation", "performance"]
    return category.replace("#", " ").replace("_", " ").lower().split()


def _flatten_doc(doc, lang: str) -> tuple[list[str], list[str], list[int], list[str], list[tuple[int, int]]]:
    """
    Flatten a Stanza Document to a single token sequence and convert sentence-local
    heads to doc-global 1-based indexing (root=0), matching `src/utils/data_converter.py`.
    """
    tokens: list[str] = []
    pos: list[str] = []
    heads: list[int] = []
    deprel: list[str] = []
    bounds: list[tuple[int, int]] = []

    for sent in doc.sentences:
        offset = len(tokens)
        for word in sent.words:
            tokens.append(word.text)
            if lang == "en":
                pos.append(word.xpos or word.upos or "")
            else:
                pos.append(word.upos or "")

            if word.head == 0:
                heads.append(0)
            else:
                heads.append(offset + word.head)

            deprel.append(word.deprel)

            if word.start_char is not None and word.end_char is not None:
                bounds.append((word.start_char, word.end_char))
            else:
                bounds.append((-1, -1))

    return tokens, pos, heads, deprel, bounds


def convert_phone_category_xml_to_json(xml_path: Path, json_path: Path, lang: str = "nl") -> int:
    """
    Convert category-level ABSA XML into the repo JSON format.
    Since these datasets do not provide explicit opinion targets/spans, aspects are
    exported as category tokens with `from/to = null`.

    Returns: number of sentence entries written.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nlp = get_stanza_pipeline(lang)
    entries: list[dict[str, Any]] = []

    for sentence in root.findall(".//sentence"):
        text_elem = sentence.find("text")
        if text_elem is None or not (text_elem.text or "").strip():
            continue

        text = text_elem.text or ""
        opinions = sentence.find("Opinions")
        if opinions is None:
            continue

        aspects: list[dict[str, Any]] = []
        for opinion in opinions.findall("Opinion"):
            category = opinion.get("category", "").strip()
            polarity = (opinion.get("polarity", "neutral") or "neutral").strip().lower()
            if not category:
                continue

            aspects.append(
                {
                    "term": _category_to_tokens(category),
                    "from": None,
                    "to": None,
                    "polarity": polarity,
                }
            )

        if not aspects:
            continue

        doc = nlp(text)
        tokens, pos, heads, deprel, _bounds = _flatten_doc(doc, lang=lang)
        if not tokens:
            continue

        entries.append(
            {
                "token": tokens,
                "pos": pos,
                "head": heads,
                "deprel": deprel,
                "aspects": aspects,
                "sent_score": [0.0] * len(tokens),
            }
        )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(entries)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "raw-sem16-other"
    out_base = base_dir / "data" / "semeval"

    # --- Restaurant term-level datasets (explicit targets/spans) ---
    dutch_rest_dir = out_base / "Restaurant_16_DU"
    turkish_rest_dir = out_base / "Restaurant_16_TU"

    dutch_rest_dir.mkdir(parents=True, exist_ok=True)
    turkish_rest_dir.mkdir(parents=True, exist_ok=True)

    # Dutch restaurants
    convert_xml_to_csv(
        raw_dir / "restaurants_dutch_training.xml",
        dutch_rest_dir / "Restaurants_Train.csv",
        language="nl",
    )
    convert_xml_to_csv(
        raw_dir / "DU_REST_SB1_TEST.xml",
        dutch_rest_dir / "Restaurants_Test.csv",
        language="nl",
    )
    convert_csv_to_json(
        input_csv_path=dutch_rest_dir / "Restaurants_Train.csv",
        output_json_path=dutch_rest_dir / "Restaurants_Train.json",
        dataset_choice=9,
    )
    convert_csv_to_json(
        input_csv_path=dutch_rest_dir / "Restaurants_Test.csv",
        output_json_path=dutch_rest_dir / "Restaurants_Test.json",
        dataset_choice=9,
    )

    # Turkish restaurants
    convert_xml_to_csv(
        raw_dir / "TU_REST_SB1_TRAINING.xml",
        turkish_rest_dir / "Restaurants_Train.csv",
        language="tr",
    )
    convert_xml_to_csv(
        raw_dir / "TU_REST_SB1_TEST.xml",
        turkish_rest_dir / "Restaurants_Test.csv",
        language="tr",
    )
    convert_csv_to_json(
        input_csv_path=turkish_rest_dir / "Restaurants_Train.csv",
        output_json_path=turkish_rest_dir / "Restaurants_Train.json",
        dataset_choice=10,
    )
    convert_csv_to_json(
        input_csv_path=turkish_rest_dir / "Restaurants_Test.csv",
        output_json_path=turkish_rest_dir / "Restaurants_Test.json",
        dataset_choice=10,
    )

    # --- Phone category-level dataset (no explicit targets/spans) ---
    dutch_phone_dir = out_base / "Phone_16_DU"
    dutch_phone_dir.mkdir(parents=True, exist_ok=True)

    n_train = convert_phone_category_xml_to_json(
        raw_dir / "smartphones_Dutch_training.xml",
        dutch_phone_dir / "Phones_Train.json",
        lang="nl",
    )
    n_test = convert_phone_category_xml_to_json(
        raw_dir / "DU_PHNS_SB1_TEST_.xml",
        dutch_phone_dir / "Phones_Test.json",
        lang="nl",
    )

    print("\n=== SemEval16-other conversion summary ===")
    print(f"- Restaurant_16_DU: {dutch_rest_dir}")
    print(f"- Restaurant_16_TU: {turkish_rest_dir}")
    print(f"- Phone_16_DU: {dutch_phone_dir} (sentences: train={n_train}, test={n_test})")


if __name__ == "__main__":
    main()

