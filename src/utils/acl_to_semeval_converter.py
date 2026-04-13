#!/usr/bin/env python3
# src/utils/acl_to_semeval_converter.py
# -*- coding: utf-8 -*-

"""
Convert ACL-14-short-data .raw format to SemEval CSV format.

ACL .raw format:
- Every 3 lines form a group:
  1. Text with $T$ placeholder
  2. Aspect term
  3. Sentiment label (-1/0/1)

SemEval CSV format:
- Columns: sentence_id,sentence_text,aspect_term,polarity,from,to
- polarity: negative/neutral/positive
- from/to: character positions of aspect term
"""

import csv
from pathlib import Path
from typing import List, Tuple, Dict


def convert_sentiment_label(label: str) -> str:
    """Convert numeric sentiment label to text."""
    label_map = {
        "-1": "negative",
        "0": "neutral", 
        "1": "positive"
    }
    return label_map.get(label.strip(), "neutral")


def find_aspect_positions(text: str, aspect_term: str) -> Tuple[int, int]:
    """
    Find the character positions of aspect term in text.
    Returns (from, to) positions.
    """
    # Replace $T$ with aspect term and find positions
    if "$T$" in text:
        # Find position of $T$ placeholder
        placeholder_start = text.find("$T$")
        if placeholder_start != -1:
            # Replace $T$ with actual aspect term
            new_text = text.replace("$T$", aspect_term)
            from_pos = placeholder_start
            to_pos = placeholder_start + len(aspect_term)
            return from_pos, to_pos
    
    # If $T$ not found, try direct search
    start_pos = text.find(aspect_term)
    if start_pos != -1:
        return start_pos, start_pos + len(aspect_term)
    
    # If not found, try case-insensitive search
    lower_text = text.lower()
    lower_aspect = aspect_term.lower()
    start_pos = lower_text.find(lower_aspect)
    if start_pos != -1:
        return start_pos, start_pos + len(aspect_term)
    
    # Default fallback (shouldn't happen in well-formed data)
    return 0, len(aspect_term)


def parse_raw_file(raw_file_path: Path) -> List[Dict[str, str]]:
    """
    Parse ACL .raw file and return list of entries.
    """
    entries = []
    
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process every 3 lines as a group
    sentence_id = 0
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
            
        text = lines[i].strip()
        aspect_term = lines[i + 1].strip()
        sentiment_label = lines[i + 2].strip()
        
        # Skip empty entries
        if not text or not aspect_term:
            continue
            
        # Convert sentiment label
        polarity = convert_sentiment_label(sentiment_label)
        
        # Replace $T$ with aspect term for final sentence text
        sentence_text = text.replace("$T$", aspect_term)
        
        # Find aspect positions in the replaced text
        from_pos, to_pos = find_aspect_positions(text, aspect_term)
        
        entries.append({
            'sentence_id': str(sentence_id),
            'sentence_text': sentence_text,
            'aspect_term': aspect_term,
            'polarity': polarity,
            'from': str(from_pos),
            'to': str(to_pos)
        })
        
        sentence_id += 1
    
    return entries


def convert_raw_to_csv(raw_file_path: Path, csv_file_path: Path):
    """
    Convert ACL .raw file to SemEval CSV format.
    """
    print(f"Converting {raw_file_path} to {csv_file_path}")
    
    # Parse raw file
    entries = parse_raw_file(raw_file_path)
    
    # Write CSV file
    fieldnames = ['sentence_id', 'sentence_text', 'aspect_term', 'polarity', 'from', 'to']
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)
    
    print(f"✅ Converted {len(entries)} entries to {csv_file_path}")


def main():
    """Main function to convert both train and test files."""
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    acl_dir = base_dir / "data" / "acl-14-short-data-csv"
    
    # Input .raw files
    train_raw = acl_dir / "train.raw"
    test_raw = acl_dir / "test.raw"
    
    # Output .csv files
    train_csv = acl_dir / "train.csv"
    test_csv = acl_dir / "test.csv"
    
    print("=== ACL to SemEval Format Converter ===\n")
    
    # Convert train file
    if train_raw.exists():
        convert_raw_to_csv(train_raw, train_csv)
    else:
        print(f"⚠️  Training file not found: {train_raw}")
    
    # Convert test file  
    if test_raw.exists():
        convert_raw_to_csv(test_raw, test_csv)
    else:
        print(f"⚠️  Test file not found: {test_raw}")
    
    print(f"\n✅ Conversion complete!")
    print(f"Files created:")
    if train_csv.exists():
        print(f"  - {train_csv}")
    if test_csv.exists():
        print(f"  - {test_csv}")


if __name__ == "__main__":
    main()