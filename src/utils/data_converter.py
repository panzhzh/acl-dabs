#!/usr/bin/env python3
# src/utils/data_converter.py
# -*- coding: utf-8 -*-

"""
Data format conversion utilities.
Convert different CSV formats to JSON format with proper SenticNet-8 token scores.

Supports multiple datasets:
1. ACL-14-short (format: sentence_id, sentence_text, aspect_term, polarity)
2. SemEval Laptop_14 (format: sentence_id, sentence_text, aspect_term, polarity, from, to)
3. SemEval Restaurant_14 (format: sentence_id, sentence_text, aspect_term, polarity, from, to)
4. SemEval Restaurant_15 (format: sentence_id, sentence_text, aspect_term, polarity, from, to)

★ 2025-05 update: switched from spaCy to **Stanza UD pipeline** so that the output
  matches Stanford-style dependencies (lower-case labels, 1-based heads, `obj/obl`
  etc.) exactly like the "second segment" sample.

Run once to download Stanza models:
    >>> import stanza
    >>> stanza.download('en')
    >>> stanza.download('nl')  # Dutch
    >>> stanza.download('tr')  # Turkish

GPU will be used automatically if CUDA is available (`use_gpu=True`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import stanza 
from tqdm import tqdm

# ============================================
# Dataset selection configuration
# ============================================
DATASET_CHOICE = 5  # Change this value to select a different dataset.

DATASET_CONFIGS = {
    1: {
        "name": "ACL-14-short",
        "train_csv": "data/acl-14-short-data-csv/train.csv",
        "test_csv": "data/acl-14-short-data-csv/test.csv",
        "has_position": True,
        "language": "en",
        "description": "ACL 2014 Short Data (Twitter sentiment)"
    },
    2: {
        "name": "SemEval Laptop_14",
        "train_csv": "data/semeval/Laptop_14/Laptops_Train.csv",
        "test_csv": "data/semeval/Laptop_14/Laptops_Test.csv",
        "has_position": True,
        "language": "en",
        "description": "SemEval 2014 Laptop Domain"
    },
    3: {
        "name": "SemEval Restaurant_14",
        "train_csv": "data/semeval/Restaurant_14/Restaurants_Train.csv",
        "test_csv": "data/semeval/Restaurant_14/Restaurants_Test.csv",
        "has_position": True,
        "language": "en",
        "description": "SemEval 2014 Restaurant Domain"
    },
    4: {
        "name": "SemEval Restaurant_15",
        "train_csv": "data/semeval/Restaurant_15/Restaurants_Train.csv",
        "test_csv": "data/semeval/Restaurant_15/Restaurants_Test.csv",
        "has_position": True,
        "language": "en",
        "description": "SemEval 2015 Restaurant Domain"
    },
    5: {
        "name": "SemEval Restaurant_16",
        "train_csv": "data/semeval/Restaurant_16/Restaurants_Train.csv",
        "test_csv": "data/semeval/Restaurant_16/Restaurants_Test.csv",
        "has_position": True,
        "language": "en",
        "description": "SemEval 2016 Restaurant Domain (English)"
    },
    # Multilingual Restaurant_16 datasets
    6: {
        "name": "SemEval Restaurant_16 (French)",
        "train_csv": "data/semeval/Restaurant_16_FR/Restaurants_Train.csv",
        "test_csv": "data/semeval/Restaurant_16_FR/Restaurants_Test.csv",
        "has_position": True,
        "language": "fr",
        "description": "SemEval 2016 Restaurant Domain (French)"
    },
    7: {
        "name": "SemEval Restaurant_16 (Russian)",
        "train_csv": "data/semeval/Restaurant_16_RU/Restaurants_Train.csv",
        "test_csv": "data/semeval/Restaurant_16_RU/Restaurants_Test.csv",
        "has_position": True,
        "language": "ru",
        "description": "SemEval 2016 Restaurant Domain (Russian)"
    },
    8: {
        "name": "SemEval Restaurant_16 (Spanish)",
        "train_csv": "data/semeval/Restaurant_16_ES/Restaurants_Train.csv",
        "test_csv": "data/semeval/Restaurant_16_ES/Restaurants_Test.csv",
        "has_position": True,
        "language": "es",
        "description": "SemEval 2016 Restaurant Domain (Spanish)"
    },
    9: {
        "name": "SemEval Restaurant_16 (Dutch)",
        "train_csv": "data/semeval/Restaurant_16_DU/Restaurants_Train.csv",
        "test_csv": "data/semeval/Restaurant_16_DU/Restaurants_Test.csv",
        "has_position": True,
        "language": "nl",
        "description": "SemEval 2016 Restaurant Domain (Dutch)"
    },
    10: {
        "name": "SemEval Restaurant_16 (Turkish)",
        "train_csv": "data/semeval/Restaurant_16_TU/Restaurants_Train.csv",
        "test_csv": "data/semeval/Restaurant_16_TU/Restaurants_Test.csv",
        "has_position": True,
        "language": "tr",
        "description": "SemEval 2016 Restaurant Domain (Turkish)"
    }
}

# ─────────────── Path Configuration ───────────────
# Use paths relative to the current file location.
CURRENT_DIR = Path(__file__).parent.parent.parent
SENTIC_XLSX = CURRENT_DIR.parent / "senticnet" / "senticnet.xlsx"  # If SenticNet data is available.

# Resolve train/test paths from the selected dataset configuration.
if DATASET_CHOICE in DATASET_CONFIGS:
    config = DATASET_CONFIGS[DATASET_CHOICE]
    TRAIN_CSV = CURRENT_DIR / config["train_csv"]
    TEST_CSV = CURRENT_DIR / config["test_csv"]
else:
    raise ValueError(f"Invalid DATASET_CHOICE: {DATASET_CHOICE}. Must be 1-5.")
# ──────────────────────────────────────

##############################################################################
# 1. Load SenticNet-8 (using only POLARITY INTENSITY)                         #
##############################################################################

def load_senticnet(path: Path) -> dict[str, float]:
    """Return mapping {full_concept: polarity_intensity}. Multi-word concepts are
    **not** split. Missing or non-numeric intensities are set to 0.0.
    """
    df = pd.read_excel(
        path,
        engine="openpyxl",
        usecols=["CONCEPT", "POLARITY INTENSITY"],
        dtype={"CONCEPT": str},
    )

    lookup: dict[str, float] = {}
    for concept, inten in zip(df["CONCEPT"], df["POLARITY INTENSITY"]):
        if pd.isna(concept):
            continue
        try:
            polarity = float(inten)
        except (TypeError, ValueError):
            polarity = 0.0

        phrase = concept.replace("_", " ").lower().strip()
        lookup[phrase] = polarity
    return lookup


# Use an empty lookup when the SenticNet file is unavailable.
sentic_dict = load_senticnet(SENTIC_XLSX) if SENTIC_XLSX.exists() else {}
DEFAULT_SCORE = 0.0

##############################################################################
# 2. Stanza Preprocessing (UD v2) - Multilingual Support                      #
##############################################################################
# Download models with:
#   stanza.download('en')  # English
#   stanza.download('fr')  # French
#   stanza.download('ru')  # Russian
#   stanza.download('es')  # Spanish

# Language code mapping
LANGUAGE_MAP = {
    'en': 'en',
    'fr': 'fr',
    'ru': 'ru',
    'es': 'es',
    'nl': 'nl',
    'tr': 'tr',
}

# Cache initialized Stanza pipelines.
_stanza_pipelines = {}

def get_stanza_pipeline(lang: str = 'en') -> stanza.Pipeline:
    """Get or create a Stanza pipeline for the requested language."""
    # Normalize the language code.
    lang_code = LANGUAGE_MAP.get(lang, lang)

    if lang_code not in _stanza_pipelines:
        print(f"Loading Stanza pipeline for language: {lang_code}")

        # Some languages do not ship an MWT model. Try with MWT first (when applicable),
        # then fall back to a no-MWT pipeline. Never silently fall back to English,
        # because that can break char-offset alignment for non-English datasets.
        if lang_code in ['ru', 'zh', 'zh-hans']:
            processor_candidates = ["tokenize,pos,lemma,depparse"]
            print(f"ℹ️  Language '{lang_code}' detected: disabling 'mwt' processor.")
        else:
            processor_candidates = [
                "tokenize,mwt,pos,lemma,depparse",
                "tokenize,pos,lemma,depparse",
            ]

        last_err = None
        for processors_list in processor_candidates:
            try:
                _stanza_pipelines[lang_code] = stanza.Pipeline(
                    lang=lang_code,
                    processors=processors_list,
                    use_gpu=True,
                    verbose=False,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e

        if last_err is not None:
            raise RuntimeError(
                f"Failed to load Stanza pipeline for '{lang_code}'. "
                f"Try downloading models first: `python -c \"import stanza; stanza.download('{lang_code}')\"`. "
                f"Original error: {last_err}"
            )

    return _stanza_pipelines[lang_code]

# Default to the English pipeline.
nlp = get_stanza_pipeline('en')

# Penn-Treebank XPOS tags we treat as neutral (zero sentiment)
NEUTRAL_POS = {
    "DT", "PRP", "IN", "CC", "SCONJ", "PART", "AUX",
    "PUNCT", "SYM", "SPACE",
}

##############################################################################
# 3. Helper Functions                                                        #
##############################################################################

def flatten_sentence(sent: stanza.models.common.doc.Sentence) -> Tuple[
    List[str], List[str], List[int], List[str], List[Tuple[int, int]]
]:
    """
    Extract token-level arrays and char bounds from a Stanza Sentence.
    - `token`  : raw token texts (same granularity as your CSV "from/to")
    - `pos`    : Penn-style XPOS (e.g. VBP, NN)
    - `head`   : 1-based head index, root→0  (matches Stanza/UD)
    - `deprel` : UD-v2 dependency labels  (root/obj/obl/…)
    - `bounds` : [(char_start, char_end)]  relative to sentence (0-based)
    """
    tokens, pos, heads, deprel, bounds = [], [], [], [], []

    for w in sent.words:                       # ← word = token here (English)
        tokens.append(w.text)
        pos.append(w.xpos or w.upos)          # fallback to UPOS if XPOS missing
        heads.append(w.head)                  # already 1-based; root = 0
        deprel.append(w.deprel)
        bounds.append((w.start_char, w.end_char))

    return tokens, pos, heads, deprel, bounds


def char2tok(bounds: List[Tuple[int, int]], char_from: int, char_to: int):
    """Map character span [from,to) to token span [start,end)."""
    covered = [i for i, (s, e) in enumerate(bounds) if not (e <= char_from or s >= char_to)]
    if not covered:
        raise ValueError(f"Failed to align character span [{char_from},{char_to})")
    return covered[0], covered[-1] + 1


def token_sent_scores(tokens: List[str], pos_tags: List[str], lexicon: dict[str, float]):
    """Assign SenticNet polarity per token via longest 3→2→1-gram match."""
    scores = [DEFAULT_SCORE] * len(tokens)
    lowered = [t.lower() for t in tokens]

    i = 0
    while i < len(tokens):
        matched = False
        for n in (3, 2, 1):
            if i + n <= len(tokens):
                phrase = " ".join(lowered[i : i + n])
                if phrase in lexicon:
                    val = lexicon[phrase]
                    for j in range(i, i + n):
                        scores[j] = val
                    i += n
                    matched = True
                    break
        if not matched:
            i += 1

    # Zero-out neutral/function words
    for idx, tag in enumerate(pos_tags):
        if tag.split("|")[0] in NEUTRAL_POS:
            scores[idx] = DEFAULT_SCORE
    return scores

##############################################################################
# 4. Main Flow                                                               #
##############################################################################

def convert_csv_to_json(input_csv_path: Path = None, output_json_path: Path = None, dataset_choice: int = None):
    """
    Convert CSV format data to JSON format. Supports multilingual data.

    Args:
        input_csv_path: Path to input CSV file (optional, uses default if None)
        output_json_path: Path to output JSON file (optional, uses default if None)
        dataset_choice: Dataset choice (1-10, optional, uses global DATASET_CHOICE if None)
    """
    # Use provided choice or global default
    choice = dataset_choice or DATASET_CHOICE
    if choice not in DATASET_CONFIGS:
        raise ValueError(f"Invalid dataset_choice: {choice}. Must be 1-10.")

    config = DATASET_CONFIGS[choice]

    # Use provided paths
    if input_csv_path is None:
        raise ValueError("input_csv_path must be provided")
    csv_path = input_csv_path
    json_path = output_json_path or csv_path.with_suffix(".json")

    # Get the language from config, default to English
    language = config.get("language", "en")
    print(f"\n=== Converting {config['name']} (Language: {language}) ===")
    print(f"Input: {csv_path}")
    print(f"Output: {json_path}")

    # 4.1 Read the CSV and filter rows.
    if not csv_path.exists():
        print(f"Error: input file does not exist: {csv_path}")
        return

    # Try different encodings
    for encoding in ['utf-8', 'gbk', 'latin-1']:
        try:
            df = pd.read_csv(csv_path, keep_default_na=False, encoding=encoding)
            print(f"Successfully read CSV with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("Failed to read CSV with any encoding")
        return

    # Filter invalid rows.
    df = df.dropna(subset=["sentence_text", "aspect_term", "polarity"])  # Drop empty rows.
    df = df[df["aspect_term"].str.strip() != ""]  # Drop empty aspect terms.
    df = df[df["aspect_term"].str.upper() != "NULL"]  # Drop NULL aspect terms.
    df = df[df["polarity"].str.lower() != "conflict"]  # Drop conflict labels.

    entries: list[dict] = []

    # Get the appropriate Stanza pipeline for the language
    nlp_lang = get_stanza_pipeline(language)

    # 4.2 Process sentence by sentence.
    for _sid, grp in tqdm(df.groupby("sentence_id"), desc="Processing"):
        text = grp["sentence_text"].iloc[0]
        doc = nlp_lang(text)

        # ---------- 1) Flatten the entire document ----------
        def flatten_doc(doc: stanza.models.common.doc.Document, lang: str = 'en') -> Tuple[
            List[str], List[str], List[int], List[str], List[Tuple[int, int]]
        ]:
            """
            Flatten a Stanza Document into one token sequence and convert sentence-local
            head indices into document-level 1-based indices (root=0). Returns:
            tokens  : str[]
            pos     : Penn-style XPOS (fallback to UPOS when unavailable)
            heads   : int[]  (document-level 1-based, root=0)
            deprel  : str[]
            bounds  : [(char_from, char_to)]  # document-level character offsets
            """
            tokens, pos, heads, deprel, bounds = [], [], [], [], []

            for sent in doc.sentences:
                offset = len(tokens)          # Starting index of this sentence in the document sequence.
                for word in sent.words:       # English: word == token
                    # 1. Token text.
                    tokens.append(word.text)

                    # 2. POS tag: use XPOS for English, UPOS for other languages.
                    if lang == 'en':
                        pos.append(word.xpos or word.upos or "")
                    else:
                        pos.append(word.upos or "")

                    # 3. Head index: sentence-local 1-based -> document-level 1-based.
                    if word.head == 0:
                        heads.append(0)                         # root
                    else:
                        heads.append(offset + word.head)        # already 1-based

                    # 4. Dependency relation label.
                    deprel.append(word.deprel)

                    # 5. Character boundaries (Stanza already uses document-level offsets).
                    if word.start_char is not None and word.end_char is not None:
                        bounds.append((word.start_char, word.end_char))
                    else:
                        # Without character offsets, char2tok will fail; record the error.
                        print(f"[ERROR] Token '{word.text}' missing char offsets.")
                        bounds.append((-1, -1))

            # Sanity check: head indices should never be negative.
            assert all(h >= 0 for h in heads), "Head indices must all be non-negative."

            return tokens, pos, heads, deprel, bounds

        # ---------- 2) Flatten the parsed document ----------
        tokens, pos, heads, deprel, bounds = flatten_doc(doc, lang=language)

        # ---- Collect aspects ----
        aspects = []
        for _, row in grp.iterrows():
            aspect_term = str(row["aspect_term"]).strip()
            polarity = str(row["polarity"]).lower()
            
            if config["has_position"]:
                # SemEval datasets provide from/to fields, so use character positions.
                try:
                    t_from, t_to = char2tok(bounds, int(row["from"]), int(row["to"]))
                    term_tokens = tokens[t_from:t_to]
                except ValueError as e:
                    print(f"[WARN] {e}; skipping this aspect")
                    continue
                except KeyError as e:
                    print(f"[WARN] Missing column {e}; skipping this aspect")
                    continue
                    
                aspects.append({
                    "term": term_tokens,
                    "from": int(t_from), 
                    "to": int(t_to),
                    "polarity": polarity,
                })
            else:
                # ACL datasets do not provide from/to fields, so locate the aspect term in the text.
                try:
                    # Find the aspect term inside the sentence text.
                    text_lower = text.lower()
                    term_lower = aspect_term.lower()
                    char_start = text_lower.find(term_lower)
                    
                    if char_start == -1:
                        # If direct matching fails, fall back to token-level matching.
                        term_words = aspect_term.split()
                        term_tokens = term_words  # Simple whitespace tokenization.
                        print(f"[INFO] Direct match failed for '{aspect_term}', using term as tokens")
                        
                        aspects.append({
                            "term": term_tokens,
                            "from": 0,  # Default fallback position.
                            "to": len(term_tokens),
                            "polarity": polarity,
                        })
                    else:
                        char_end = char_start + len(aspect_term)
                        t_from, t_to = char2tok(bounds, char_start, char_end)
                        term_tokens = tokens[t_from:t_to]
                        
                        aspects.append({
                            "term": term_tokens,
                            "from": int(t_from),
                            "to": int(t_to), 
                            "polarity": polarity,
                        })
                        
                except Exception as e:
                    print(f"[WARN] Error processing aspect '{aspect_term}': {e}")
                    # Fallback: use the raw aspect term tokens directly.
                    aspects.append({
                        "term": aspect_term.split(),
                        "from": 0,
                        "to": len(aspect_term.split()),
                        "polarity": polarity,
                    })

        if not aspects:
            continue

        # Only compute SenticNet scores for English; use zeros for other languages.
        if language == 'en':
            sent_score = token_sent_scores(tokens, pos, sentic_dict)
        else:
            sent_score = [0.0] * len(tokens)

        entries.append(
            {
                "token": tokens,
                "pos": pos,
                "head": heads,
                "deprel": deprel,
                "aspects": aspects,
                "sent_score": sent_score,
            }
        )

    # 4.3 Write JSON output.
    json_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✅ Done. Wrote {len(entries)} sentences to {json_path}")


def convert_dataset(dataset_choice: int = None):
    """Convert both train and test files for a dataset."""
    choice = dataset_choice or DATASET_CHOICE
    if choice not in DATASET_CONFIGS:
        raise ValueError(f"Invalid dataset_choice: {choice}. Must be 1-10.")

    config = DATASET_CONFIGS[choice]
    train_csv = CURRENT_DIR / config["train_csv"]
    test_csv = CURRENT_DIR / config["test_csv"]

    print(f"\n=== Converting {config['name']} ===")

    # Convert train set
    if train_csv.exists():
        print(f"\n--- Converting Training Set ---")
        convert_csv_to_json(input_csv_path=train_csv, dataset_choice=choice)
    else:
        print(f"⚠️  Training file not found: {train_csv}")

    # Convert test set
    if test_csv.exists():
        print(f"\n--- Converting Test Set ---")
        convert_csv_to_json(input_csv_path=test_csv, dataset_choice=choice)
    else:
        print(f"⚠️  Test file not found: {test_csv}")

def main():
    """Main function for command line usage."""
    print("=== Data Converter for ABSA Datasets ===")
    print("\nAvailable datasets:")
    for i, config in DATASET_CONFIGS.items():
        print(f"  {i}. {config['name']} - {config['description']}")

    print(f"\nCurrent selection: {DATASET_CHOICE} ({DATASET_CONFIGS[DATASET_CHOICE]['name']})")
    print("To change dataset, modify DATASET_CHOICE variable at the top of this file.")
    print("Or pass dataset_choice parameter to convert_dataset(choice) function.\n")

    # Convert both train and test
    convert_dataset()


if __name__ == "__main__":
    main()
