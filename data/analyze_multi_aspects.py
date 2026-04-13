#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter

def analyze_multi_aspects():
    """Analyze sentence samples that contain multiple aspects."""
    
    file_path = "/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_14/Restaurants_Train.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total sentences: {len(data)}")
    
    # Collect sentences that contain multiple aspects.
    multi_aspect_sentences = []
    
    for item in data:
        if len(item['aspects']) > 1:
            multi_aspect_sentences.append(item)
    
    print(f"Sentences with multiple aspects: {len(multi_aspect_sentences)}")
    
    # Analyze the polarity distribution.
    polarity_patterns = Counter()
    
    # Collect interesting samples for detailed analysis.
    interesting_samples = []
    
    for item in multi_aspect_sentences:
        # Reconstruct the sentence text.
        sentence = ' '.join(item['token'])
        aspects = item['aspects']
        
        # Count polarity patterns.
        polarities = [asp['polarity'] for asp in aspects]
        polarity_pattern = '-'.join(sorted(polarities))
        polarity_patterns[polarity_pattern] += 1
        
        # Keep representative samples.
        if len(aspects) >= 2:
            # Gather aspect span information.
            aspect_info = []
            for asp in aspects:
                aspect_info.append({
                    'term': ' '.join(asp['term']) if isinstance(asp['term'], list) else asp['term'],
                    'polarity': asp['polarity'],
                    'from': asp['from'],
                    'to': asp['to']
                })
            
            # Sort aspects by position.
            aspect_info.sort(key=lambda x: x['from'])
            
            interesting_samples.append({
                'sentence': sentence,
                'aspects': aspect_info,
                'polarities': polarities
            })
    
    print("\nPolarity pattern statistics:")
    for pattern, count in polarity_patterns.most_common():
        print(f"{pattern}: {count} occurrences")
    
    # Inspect representative samples in detail.
    print("\n=== Detailed Sample Analysis ===")
    
    # Group samples by polarity pattern.
    samples_by_pattern = {}
    for sample in interesting_samples:
        pattern = '-'.join(sorted(sample['polarities']))
        if pattern not in samples_by_pattern:
            samples_by_pattern[pattern] = []
        samples_by_pattern[pattern].append(sample)
    
    sample_count = 0
    for pattern in ['negative-positive', 'negative-negative', 'positive-positive', 'negative-neutral', 'positive-neutral']:
        if pattern in samples_by_pattern and sample_count < 10:
            print(f"\n--- {pattern.upper()} Pattern Samples ---")
            for i, sample in enumerate(samples_by_pattern[pattern][:3]):
                if sample_count >= 10:
                    break
                sample_count += 1
                print(f"\nSample {sample_count}:")
                print(f"Sentence: {sample['sentence']}")
                print("Aspects:")
                for j, asp in enumerate(sample['aspects']):
                    print(f"  {j+1}. '{asp['term']}' - {asp['polarity']} [span: {asp['from']}-{asp['to']}]")
                
                # Analyze the positional relationship.
                if len(sample['aspects']) >= 2:
                    asp1, asp2 = sample['aspects'][0], sample['aspects'][1]
                    distance = asp2['from'] - asp1['to']
                    print(f"  Position relation: gap = {distance}")
                
                # Analyze the polarity relationship.
                unique_polarities = list(set(sample['polarities']))
                if len(unique_polarities) > 1:
                    print(f"  Polarity relation: polarity contrast exists ({', '.join(unique_polarities)})")
                else:
                    print(f"  Polarity relation: consistent polarity ({unique_polarities[0]})")
    
    return interesting_samples

if __name__ == "__main__":
    samples = analyze_multi_aspects()
