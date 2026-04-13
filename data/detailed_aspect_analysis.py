#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import defaultdict

def analyze_logical_relationships():
    """Analyze logical relationships between aspects in more detail."""
    
    file_path = "/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_14/Restaurants_Train.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect sentences that contain multiple aspects.
    multi_aspect_sentences = [item for item in data if len(item['aspects']) > 1]
    
    print("=== Logical Relationship Analysis for Multi-Aspect Sentences ===")
    print(f"Found {len(multi_aspect_sentences)} sentences containing multiple aspects\n")
    
    # Logical marker vocabularies.
    contrast_words = ['but', 'however', 'although', 'though', 'while', 'whereas', 'yet', 'nevertheless', 'nonetheless']
    causal_words = ['because', 'since', 'due to', 'as a result', 'therefore', 'thus', 'so', 'consequently']
    additive_words = ['and', 'also', 'moreover', 'furthermore', 'additionally', 'plus', 'as well as']
    
    # Analysis buckets.
    logical_patterns = {
        'contrast': [],
        'causal': [],
        'additive': [],
        'mixed_polarity': [],
        'same_polarity': []
    }
    
    for item in multi_aspect_sentences:
        sentence = ' '.join(item['token']).lower()
        aspects = item['aspects']
        
        # Extract aspect information.
        aspect_details = []
        for asp in aspects:
            term = ' '.join(asp['term']) if isinstance(asp['term'], list) else asp['term']
            aspect_details.append({
                'term': term,
                'polarity': asp['polarity'],
                'from': asp['from'],
                'to': asp['to'],
                'position_text': ' '.join(item['token'][asp['from']:asp['to']])
            })
        
        # Sort by position.
        aspect_details.sort(key=lambda x: x['from'])
        
        # Analyze polarity relationships.
        polarities = [asp['polarity'] for asp in aspect_details]
        unique_polarities = set(polarities)
        
        sample_info = {
            'sentence': ' '.join(item['token']),
            'aspects': aspect_details,
            'polarities': polarities
        }
        
        # Detect logical marker words.
        has_contrast = any(word in sentence for word in contrast_words)
        has_causal = any(word in sentence for word in causal_words)
        has_additive = any(word in sentence for word in additive_words)
        
        # Bucket the sample.
        if has_contrast:
            logical_patterns['contrast'].append(sample_info)
        if has_causal:
            logical_patterns['causal'].append(sample_info)
        if has_additive:
            logical_patterns['additive'].append(sample_info)
        
        if len(unique_polarities) > 1:
            logical_patterns['mixed_polarity'].append(sample_info)
        else:
            logical_patterns['same_polarity'].append(sample_info)
    
    # Print aggregate results.
    print("=== Logical Pattern Statistics ===")
    print(f"Contrast relations (but, however, although, etc.): {len(logical_patterns['contrast'])}")
    print(f"Causal relations (because, since, therefore, etc.): {len(logical_patterns['causal'])}")
    print(f"Additive relations (and, also, etc.): {len(logical_patterns['additive'])}")
    print(f"Mixed polarity: {len(logical_patterns['mixed_polarity'])}")
    print(f"Same polarity: {len(logical_patterns['same_polarity'])}")
    
    # Detailed example analysis.
    print("\n=== Representative Example Analysis ===\n")
    
    # 1. Contrast samples
    print("1. Contrast samples (containing but, however, although, etc.)")
    print("-" * 60)
    for i, sample in enumerate(logical_patterns['contrast'][:5]):
        print(f"\nSample {i+1}:")
        print(f"Sentence: {sample['sentence']}")
        print("Aspects:")
        for j, asp in enumerate(sample['aspects']):
            print(f"  - '{asp['term']}' ({asp['polarity']}) span: {asp['from']}-{asp['to']}")
        
        # Analyze contrast polarity.
        polarities = [asp['polarity'] for asp in sample['aspects']]
        if len(set(polarities)) > 1:
            print(f"  ✓ Polarity contrast present: {' vs '.join(set(polarities))}")
        else:
            print(f"  - Same polarity: {polarities[0]}")
        
        # Identify the specific contrast markers.
        sentence_lower = sample['sentence'].lower()
        found_contrasts = [word for word in contrast_words if word in sentence_lower]
        print(f"  Contrast markers: {', '.join(found_contrasts)}")
    
    # 2. Causal samples
    print("\n\n2. Causal samples (containing because, since, therefore, etc.)")
    print("-" * 60)
    for i, sample in enumerate(logical_patterns['causal'][:3]):
        print(f"\nSample {i+1}:")
        print(f"Sentence: {sample['sentence']}")
        print("Aspects:")
        for j, asp in enumerate(sample['aspects']):
            print(f"  - '{asp['term']}' ({asp['polarity']}) span: {asp['from']}-{asp['to']}")
        
        sentence_lower = sample['sentence'].lower()
        found_causals = [word for word in causal_words if word in sentence_lower]
        print(f"  Causal markers: {', '.join(found_causals)}")
    
    # 3. Polarity contrast analysis
    print("\n\n3. Polarity Contrast Analysis")
    print("-" * 60)
    
    # Count polarity contrast patterns.
    polarity_patterns = defaultdict(int)
    for sample in logical_patterns['mixed_polarity']:
        pattern = '-'.join(sorted(set(sample['polarities'])))
        polarity_patterns[pattern] += 1
    
    print("Polarity contrast frequencies:")
    for pattern, count in sorted(polarity_patterns.items()):
        print(f"  {pattern}: {count} occurrences")
    
    # 4. Positional relationship analysis
    print("\n\n4. Aspect Positional Relationship Analysis")
    print("-" * 60)
    
    distances = []
    for sample in multi_aspect_sentences[:200]:  # Analyze the first 200 samples.
        aspects = sample['aspects']
        if len(aspects) >= 2:
            # Compute distances between neighboring aspects.
            for i in range(len(aspects) - 1):
                asp1 = aspects[i]
                asp2 = aspects[i + 1]
                # Ensure ascending order by position.
                if asp1['from'] > asp2['from']:
                    asp1, asp2 = asp2, asp1
                distance = asp2['from'] - asp1['to']
                distances.append(distance)
    
    if distances:
        avg_distance = sum(distances) / len(distances)
        print(f"Average aspect gap: {avg_distance:.2f} tokens")
        print(f"Minimum gap: {min(distances)} tokens")
        print(f"Maximum gap: {max(distances)} tokens")
        
        # Distance distribution.
        distance_ranges = {'0-2': 0, '3-5': 0, '6-10': 0, '11-20': 0, '20+': 0}
        for d in distances:
            if d <= 2:
                distance_ranges['0-2'] += 1
            elif d <= 5:
                distance_ranges['3-5'] += 1
            elif d <= 10:
                distance_ranges['6-10'] += 1
            elif d <= 20:
                distance_ranges['11-20'] += 1
            else:
                distance_ranges['20+'] += 1
        
        print("Distance distribution:")
        for range_name, count in distance_ranges.items():
            percentage = (count / len(distances)) * 100
            print(f"  {range_name} tokens: {count} ({percentage:.1f}%)")
    
    # 5. Special logical-pattern samples
    print("\n\n5. Special Logical-Pattern Samples")
    print("-" * 60)
    
    # Find samples that contain both contrast markers and polarity contrast.
    special_samples = []
    for sample in logical_patterns['contrast']:
        if len(set(sample['polarities'])) > 1:  # Different polarities
            special_samples.append(sample)
    
    print(f"Samples containing both contrast markers and polarity contrast: {len(special_samples)}")
    
    for i, sample in enumerate(special_samples[:3]):
        print(f"\nSpecial sample {i+1}:")
        print(f"Sentence: {sample['sentence']}")
        print("Aspects:")
        for j, asp in enumerate(sample['aspects']):
            print(f"  - '{asp['term']}' ({asp['polarity']})")
        
        # Explain the logical pattern.
        sentence_lower = sample['sentence'].lower()
        found_contrasts = [word for word in contrast_words if word in sentence_lower]
        print(f"  Logical pattern: contrast markers '{', '.join(found_contrasts)}' + polarity contrast {' vs '.join(set(sample['polarities']))}")

if __name__ == "__main__":
    analyze_logical_relationships()
