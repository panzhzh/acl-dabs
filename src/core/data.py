#!/usr/bin/env python3
# src/core/data.py
# -*- coding: utf-8 -*-

"""
Data processing and dataset handling module.
Combines data preprocessing and PyTorch dataset functionality.
"""

import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import torch
from torch.utils.data import Dataset

from .. import config

# Initialize the lemmatizer and stop-word list.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---------- Dataset Path Mapping ----------
DATASET_INFO = {
    1: dict(dir="acl-14-short-data-csv",
            train="train.json",
            test="test.json"),
    2: dict(dir=os.path.join("semeval", "Laptop_14"),
            train="Laptops_Train.json",
            test="Laptops_Test.json"),
    3: dict(dir=os.path.join("semeval", "Restaurant_14"),
            train="Restaurants_Train.json",
            test="Restaurants_Test.json"),
    4: dict(dir=os.path.join("semeval", "Restaurant_15"),
            train="Restaurants_Train.json",
            test="Restaurants_Test.json"),
    5: dict(dir=os.path.join("semeval", "Restaurant_16"),
            train="Restaurants_Train.json",
            test="Restaurants_Test.json"),
    # --- Multilingual Datasets ---
    6: dict(dir=os.path.join("semeval", "Restaurant_16_FR"),
            train="Restaurants_Train.json",
            test="Restaurants_Test.json"),
    7: dict(dir=os.path.join("semeval", "Restaurant_16_RU"),
            train="Restaurants_Train.json",
            test="Restaurants_Test.json"),
    8: dict(dir=os.path.join("semeval", "Restaurant_16_ES"),
            train="Restaurants_Train.json",
            test="Restaurants_Test.json"),
    9: dict(dir=os.path.join("semeval", "Restaurant_16_DU"),
            train="Restaurants_Train.json",
            test="Restaurants_Test.json"),
    10: dict(dir=os.path.join("semeval", "Restaurant_16_TU"),
             train="Restaurants_Train.json",
             test="Restaurants_Test.json"),
    11: dict(dir=os.path.join("semeval", "Phone_16_DU"),
             train="Phones_Train.json",
             test="Phones_Test.json"),
    # --- Cross-Dataset Combinations ---
    101: dict(dir=os.path.join("semeval", "cross_dataset_combinations"),
              train="combination_101_train.json",
              test="combination_101_test.json"),
    102: dict(dir=os.path.join("semeval", "cross_dataset_combinations"),
              train="combination_102_train.json",
              test="combination_102_test.json"),
    # --- Cross-Dataset Combinations (Updated 200-series subsets) ---
    202: dict(dir=os.path.join("semeval", "cross_dataset_combinations"),
              train="combination_202_train.json",
              test="combination_202_test.json"),
    204: dict(dir=os.path.join("semeval", "cross_dataset_combinations"),
              train="combination_204_train.json",
              test="combination_204_test.json"),
    207: dict(dir=os.path.join("semeval", "cross_dataset_combinations"),
              train="combination_207_train.json",
              test="combination_207_test.json"),
}

def parse_json_file(path):
    """
    Read JSON data shaped like:
      [{ "token": [...], "aspects": [...] }, ...]
    and flatten it into a CSV-like table with columns:
        sentence_text | aspect_term | polarity | from | to
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        json_list = json.load(f)  # Each element stores one sentence and its aspects.

    for item in json_list:
        tokens  = item["token"]
        sent    = " ".join(tokens)

        for asp in item.get("aspects", []):
            data.append({
                # Fields required for model training.
                "sentence_text": sent,
                "aspect_term"  : " ".join(asp.get("term", [])),
                "polarity"     : asp.get("polarity", ""),
                "from"         : asp.get("from", None),
                "to"           : asp.get("to", None),

                # Preserve the remaining original fields as well.
                "token"     : tokens,
                "pos"       : item.get("pos", []),
                "head"      : item.get("head", []),
                "deprel"    : item.get("deprel", []),
                "sent_score": item.get("sent_score", [])
            })

    # Convert to a DataFrame.
    df = pd.DataFrame(data)
    return df

def read_dataset(n: int = None, test_mode: bool = False):
    if n is None:
        n = config.DATASET_CHOICE
    if n not in DATASET_INFO:
        raise ValueError(f"DATASET_CHOICE {n} is not valid. Must be in {list(DATASET_INFO.keys())}")

    info = DATASET_INFO[n].copy()
    # Append the augmentation suffix to the train filename when enabled.
    if config.USE_DATA_AUG:
        root, ext = os.path.splitext(info["train"])
        info["train"] = f"{root}{config.DATA_AUG_SUFFIX}{ext}"

    train_path = config.DATA_DIR / info["dir"] / info["train"]
    test_path  = config.DATA_DIR / info["dir"] / info["test"]

    # Read the datasets through the JSON parser.
    df_train = parse_json_file(train_path)
    df_test  = parse_json_file(test_path)

    # In test mode, only keep 1% of the data.
    if test_mode:
        print(f"  [TEST MODE] Before sampling: train={len(df_train)}, test={len(df_test)}")
        df_train = df_train.sample(frac=0.01, random_state=42)
        df_test = df_test.sample(frac=0.01, random_state=42)
        print(f"  [TEST MODE] After sampling: train={len(df_train)}, test={len(df_test)}")

    # Note: do not deduplicate here.
    # Official SemEval splits count aspect instances as samples; deduplicating during
    # loading would change the sample count and break alignment with evaluation/error analysis.

    # --- Column renaming ---
    rename_map = {'sentence_text': 'text',
                'polarity'     : 'aspect_polarity'}
    df_train.rename(columns=rename_map, inplace=True)
    df_test.rename(columns=rename_map,  inplace=True)

    # Keep the legacy renaming path for compatibility.
    df_train.rename(columns={'sentence_text': 'text',
                             'polarity': 'aspect_polarity'}, inplace=True)
    df_test.rename(columns={'sentence_text': 'text',
                            'polarity': 'aspect_polarity'}, inplace=True)

    return df_train, df_test

# Aspect-marking is disabled; use the original text directly.

def preprocess_and_filter(df: pd.DataFrame):
    """
    Clean the DataFrame, drop missing values, remove conflict labels, and prepare fields.
    """
    # Drop missing values.
    df.dropna(subset=['text', 'aspect_term', 'aspect_polarity'], inplace=True)
    # Remove conflict labels.
    df = df[df['aspect_polarity'] != 'conflict']
    
    # Use the original text without inserting explicit aspect markers.
    df['clean_text'] = df['text']
    
    # Map sentiment labels to numeric IDs.
    df['label'] = df['aspect_polarity'].map(config.LABEL_MAPPING)
    
    # Ensure aspect position columns are preserved
    if 'from' not in df.columns:
        df['from'] = None
    if 'to' not in df.columns:
        df['to'] = None
    if 'token' not in df.columns:
        df['token'] = None
        
    return df

# ============================================
# PyTorch Dataset Classes
# ============================================

class TweetsDataset(Dataset):
    def __init__(self, encodings, labels, aspect_positions=None):
        self.encodings = encodings
        self.labels = labels
        self.aspect_positions = aspect_positions  # List of (start, end) tuples or None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        
        # Add aspect token positions if available
        if self.aspect_positions is not None and idx < len(self.aspect_positions):
            # Convert to list of start/end positions for dual-layer model
            aspect_pos = self.aspect_positions[idx]
            if aspect_pos is not None:
                item['aspect_token_positions'] = aspect_pos
                # Build aspect mask in tokenizer space: [seq_len]
                seq_len = item['input_ids'].shape[0]
                a_mask = torch.zeros(seq_len, dtype=torch.long)
                start, end = aspect_pos
                # Clamp to valid range and skip specials handled by tokenizer
                start = max(0, min(int(start), seq_len))
                end = max(start, min(int(end), seq_len))
                if end > start:
                    a_mask[start:end] = 1
                item['aspect_mask'] = a_mask
            else:
                item['aspect_token_positions'] = None
                # No mask available
                seq_len = item['input_ids'].shape[0]
                item['aspect_mask'] = torch.zeros(seq_len, dtype=torch.long)
        else:
            item['aspect_token_positions'] = None
            seq_len = item['input_ids'].shape[0]
            item['aspect_mask'] = torch.zeros(seq_len, dtype=torch.long)
            
        return item

def convert_token_positions_to_tokenizer_positions(tokenizer, original_tokens, aspect_from, aspect_to, tokenized_text):
    """
    Convert token-level positions (from/to) to tokenizer subword positions.
    
    Args:
        tokenizer: The tokenizer used for encoding
        original_tokens: List of original tokens from JSON
        aspect_from: Start position in original tokens
        aspect_to: End position in original tokens
        tokenized_text: The full text that was tokenized
    
    Returns:
        (start_pos, end_pos) in tokenizer space, or None if conversion fails
    """
    try:
        # Get the aspect text from original tokens
        aspect_tokens = original_tokens[aspect_from:aspect_to]
        aspect_text = " ".join(aspect_tokens)
        
        # Find the aspect text in the tokenized text (which is the clean text)
        aspect_start_char = tokenized_text.lower().find(aspect_text.lower())
        if aspect_start_char == -1:
            # Try without case sensitivity and with different spacing
            aspect_text_no_space = aspect_text.replace(" ", "")
            text_no_space = tokenized_text.replace(" ", "")
            char_pos_no_space = text_no_space.lower().find(aspect_text_no_space.lower())
            if char_pos_no_space != -1:
                # Convert back to original text position (approximate)
                aspect_start_char = char_pos_no_space
            else:
                return None
            
        aspect_end_char = aspect_start_char + len(aspect_text)
        
        # Tokenize the full text and get character-to-token mapping
        encoding = tokenizer(tokenized_text, return_offsets_mapping=True, add_special_tokens=True)
        offset_mapping = encoding['offset_mapping']
        
        # Find tokenizer positions that overlap with aspect character span
        start_token_pos = None
        end_token_pos = None
        
        for i, (start_char, end_char) in enumerate(offset_mapping):
            # Skip special tokens (CLS, SEP, PAD) which have (0,0) mapping
            if start_char == 0 and end_char == 0:
                continue
                
            # Find start position - first token that overlaps with aspect start
            if start_token_pos is None and end_char > aspect_start_char:
                start_token_pos = i
                
            # Find end position - last token that overlaps with aspect
            if start_char < aspect_end_char:
                end_token_pos = i + 1
        
        if start_token_pos is not None and end_token_pos is not None and start_token_pos < end_token_pos:
            return (start_token_pos, end_token_pos)
        else:
            return None
            
    except Exception as e:
        # If conversion fails, return None to fall back to CLS token
        return None

def tokenize_texts(tokenizer, texts, max_length=128):
    """
    Tokenize text with the given tokenizer and return dataset-ready encodings.
    """
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length
    )

def tokenize_texts_with_aspect_positions(tokenizer, texts, aspect_data, max_length=128):
    """
    Tokenize texts and convert aspect positions to tokenizer space.
    
    Args:
        tokenizer: The tokenizer to use
        texts: Series of text strings
        aspect_data: DataFrame with 'from', 'to', and 'token' columns
        max_length: Maximum sequence length
    
    Returns:
        (encodings, aspect_positions) where aspect_positions is a list of (start, end) tuples
    """
    # Standard tokenization
    encodings = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Convert aspect positions
    aspect_positions = []
    
    # Use iloc and column access instead of itertuples to avoid keyword issues
    for i, text in enumerate(texts):
        if i < len(aspect_data):
            # Access columns directly to avoid 'from' keyword issue in itertuples
            aspect_from = aspect_data.iloc[i]['from'] if 'from' in aspect_data.columns else None
            aspect_to = aspect_data.iloc[i]['to'] if 'to' in aspect_data.columns else None
            original_tokens = aspect_data.iloc[i]['token'] if 'token' in aspect_data.columns else None
            
            if (aspect_from is not None and aspect_to is not None and 
                original_tokens is not None and len(original_tokens) > 0):
                
                # Convert positions
                converted_pos = convert_token_positions_to_tokenizer_positions(
                    tokenizer, original_tokens, aspect_from, aspect_to, text
                )
                aspect_positions.append(converted_pos)
            else:
                aspect_positions.append(None)
        else:
            aspect_positions.append(None)
    
    return encodings, aspect_positions
