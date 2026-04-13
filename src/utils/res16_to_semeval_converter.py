#!/usr/bin/env python3
# src/utils/res16_to_semeval_converter.py
# -*- coding: utf-8 -*-

"""
Convert Restaurant_16 XML format to SemEval CSV format.
Supports both English and multilingual datasets.

Restaurant_16 XML format:
- <Reviews> contains multiple <Review> elements
- Each <Review> contains <sentences> with multiple <sentence> elements
- Each <sentence> has <text> and <Opinions> with <Opinion> elements
- <Opinion> has attributes: target, category, polarity, from, to

SemEval CSV format:
- Columns: sentence_id,sentence_text,aspect_term,polarity,from,to
- polarity: negative/neutral/positive
- from/to: character positions of aspect term
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional


def parse_xml_file(xml_file_path: Path, language: str = 'en') -> List[Dict[str, str]]:
    """
    Parse Restaurant_16 XML file and return list of entries.

    Args:
        xml_file_path: Path to XML file
        language: Language code (en, fr, ru, es, etc.)
    """
    entries = []

    # Parse XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Process each review
    for review in root.findall('.//Review'):
        sentences = review.find('sentences')
        if sentences is None:
            continue

        # Process each sentence
        for sentence in sentences.findall('sentence'):
            sentence_id = sentence.get('id', '')
            text_elem = sentence.find('text')
            if text_elem is None:
                continue

            sentence_text = text_elem.text or ''

            # Skip empty sentences
            if not sentence_text.strip():
                continue

            # Process opinions in this sentence
            opinions = sentence.find('Opinions')
            if opinions is None:
                continue

            for opinion in opinions.findall('Opinion'):
                target = opinion.get('target', '')
                polarity = opinion.get('polarity', 'neutral')
                from_pos = opinion.get('from', '0')
                to_pos = opinion.get('to', '0')

                # Skip NULL targets (implicit aspects)
                if target == 'NULL':
                    continue

                entries.append({
                    'sentence_id': sentence_id,
                    'sentence_text': sentence_text,
                    'aspect_term': target,
                    'polarity': polarity,
                    'from': from_pos,
                    'to': to_pos
                })

    return entries


def convert_xml_to_csv(xml_file_path: Path, csv_file_path: Path, language: str = 'en'):
    """
    Convert Restaurant_16 XML file to SemEval CSV format.

    Args:
        xml_file_path: Path to input XML file
        csv_file_path: Path to output CSV file
        language: Language code
    """
    print(f"Converting {language.upper()}: {xml_file_path.name} to {csv_file_path.name}")

    # Parse XML file
    entries = parse_xml_file(xml_file_path, language)

    # Write CSV file
    fieldnames = ['sentence_id', 'sentence_text', 'aspect_term', 'polarity', 'from', 'to']

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)

    print(f"✅ Converted {len(entries)} entries to {csv_file_path}")


def main():
    """Main function to convert both English and multilingual datasets."""
    base_dir = Path(__file__).parent.parent.parent

    # English Restaurant_16 (from semeval)
    print("=== Restaurant_16 to SemEval Format Converter ===\n")
    print("Converting English (SemEval 2016)...")
    res16_dir = base_dir / "data" / "semeval" / "Restaurant_16"

    train_xml = res16_dir / "Restaurants_Train.xml"
    test_xml = res16_dir / "Restaurants_Test.xml"
    train_csv = res16_dir / "Restaurants_Train.csv"
    test_csv = res16_dir / "Restaurants_Test.csv"

    if train_xml.exists():
        convert_xml_to_csv(train_xml, train_csv, language='en')
    else:
        print(f"⚠️  Training file not found: {train_xml}")

    if test_xml.exists():
        convert_xml_to_csv(test_xml, test_csv, language='en')
    else:
        print(f"⚠️  Test file not found: {test_xml}")

    # Multilingual datasets (from multilingual directory)
    print("\n" + "="*60)
    print("Converting Multilingual datasets...")
    print("="*60 + "\n")

    multilingual_dir = base_dir / "data" / "multilingual"

    # Define language mappings: (xml_prefix, lang_code, lang_name)
    languages = [
        ('16FR', 'fr', 'French'),
        ('16RU', 'ru', 'Russian'),
        ('16SP', 'es', 'Spanish'),
    ]

    for xml_prefix, lang_code, lang_name in languages:
        # Create output directory
        output_dir = base_dir / "data" / "semeval" / f"Restaurant_16_{lang_code.upper()}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process train file
        train_xml = multilingual_dir / f"{xml_prefix}_Restaurants_Train.xml"
        train_csv = output_dir / "Restaurants_Train.csv"

        if train_xml.exists():
            print(f"\n{lang_name} - Training data:")
            convert_xml_to_csv(train_xml, train_csv, language=lang_code)
        else:
            print(f"⚠️  {lang_name} training file not found: {train_xml}")

        # Process test file
        test_xml = multilingual_dir / f"{xml_prefix}_Restaurants_Test.xml"
        test_csv = output_dir / "Restaurants_Test.csv"

        if test_xml.exists():
            print(f"{lang_name} - Test data:")
            convert_xml_to_csv(test_xml, test_csv, language=lang_code)
        else:
            print(f"⚠️  {lang_name} test file not found: {test_xml}")

    print(f"\n{'='*60}")
    print("✅ All conversions complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()