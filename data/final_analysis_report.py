#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

def generate_final_report():
    """Generate the final relationship analysis report for multi-aspect sentences."""
    
    file_path = "/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_14/Restaurants_Train.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect sentences that contain multiple aspects.
    multi_aspect_sentences = [item for item in data if len(item['aspects']) > 1]
    
    print("=" * 80)
    print("Restaurant_Train.json Multi-Aspect Relationship Analysis Report")
    print("=" * 80)
    
    print("\n📊 Basic Statistics:")
    print(f"  • Total sentences: {len(data)}")
    print(f"  • Multi-aspect sentences: {len(multi_aspect_sentences)} ({len(multi_aspect_sentences)/len(data)*100:.1f}%)")
    
    # Select up to 10 representative samples for deeper analysis.
    selected_samples = []
    
    # 1. Contrast relations combined with polarity contrast.
    contrast_words = ['but', 'however', 'although', 'though', 'while', 'whereas', 'yet']
    
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
                'to': asp['to']
            })
        
        aspect_details.sort(key=lambda x: x['from'])
        polarities = [asp['polarity'] for asp in aspect_details]
        
        sample_info = {
            'sentence': ' '.join(item['token']),
            'aspects': aspect_details,
            'polarities': polarities,
            'type': ''
        }
        
        # Select representative samples by pattern.
        has_contrast = any(word in sentence for word in contrast_words)
        has_mixed_polarity = len(set(polarities)) > 1
        
        if has_contrast and has_mixed_polarity and len(selected_samples) < 10:
            sample_info['type'] = 'contrast + polarity contrast'
            found_contrasts = [word for word in contrast_words if word in sentence]
            sample_info['contrast_words'] = found_contrasts
            selected_samples.append(sample_info)
        elif has_contrast and not has_mixed_polarity and len([s for s in selected_samples if s['type'] == 'contrast + same polarity']) < 2:
            sample_info['type'] = 'contrast + same polarity'
            found_contrasts = [word for word in contrast_words if word in sentence]
            sample_info['contrast_words'] = found_contrasts
            selected_samples.append(sample_info)
        elif has_mixed_polarity and not has_contrast and len([s for s in selected_samples if s['type'] == 'polarity contrast']) < 3:
            sample_info['type'] = 'polarity contrast'
            selected_samples.append(sample_info)
        elif not has_contrast and not has_mixed_polarity and len([s for s in selected_samples if s['type'] == 'same polarity']) < 2:
            sample_info['type'] = 'same polarity'
            selected_samples.append(sample_info)
    
    print(f"\n🔍 Representative Sample Analysis ({len(selected_samples)} total):")
    print("=" * 80)
    
    for i, sample in enumerate(selected_samples):
        print(f"\nSample {i+1}: {sample['type']}")
        print("-" * 60)
        print(f"Sentence: {sample['sentence']}")
        print(f"Aspects: {len(sample['aspects'])}")
        
        for j, asp in enumerate(sample['aspects']):
            print(f"  {j+1}. '{asp['term']}' -> {asp['polarity']} (span: {asp['from']}-{asp['to']})")
        
        # Analyze the relationship pattern.
        print("\n🔗 Relationship Pattern Analysis:")
        
        # 1. Polarity relation
        unique_polarities = list(set(sample['polarities']))
        if len(unique_polarities) > 1:
            print(f"  • Polarity contrast: {' vs '.join(unique_polarities)}")
        else:
            print(f"  • Same polarity: {unique_polarities[0]}")
        
        # 2. Positional relation
        if len(sample['aspects']) >= 2:
            distances = []
            for k in range(len(sample['aspects']) - 1):
                asp1, asp2 = sample['aspects'][k], sample['aspects'][k+1]
                distance = asp2['from'] - asp1['to']
                distances.append(distance)
            avg_distance = sum(distances) / len(distances)
            print(f"  • Average gap: {avg_distance:.1f} tokens")
        
        # 3. Logical relation
        if 'contrast_words' in sample:
            print(f"  • Contrast logic: contains '{', '.join(sample['contrast_words'])}'")
        
        # 4. Inferred relation type
        print("  • Inferred relation type: ", end="")
        if sample['type'] == 'contrast + polarity contrast':
            print("contrastive reversal (negative then positive, or vice versa)")
        elif sample['type'] == 'contrast + same polarity':
            print("supplementary explanation (different phrasings of the same stance)")
        elif sample['type'] == 'polarity contrast':
            print("sentiment divergence (multiple evaluations inside one sentence)")
        elif sample['type'] == 'same polarity':
            print("sentiment aggregation (stacking same-direction sentiment)")
        
        print()
    
    # Aggregate statistics.
    print("\n📈 Relationship Pattern Statistics:")
    print("=" * 80)
    
    # Polarity combination statistics.
    from collections import Counter
    polarity_patterns = Counter()
    position_stats = []
    
    for item in multi_aspect_sentences:
        aspects = item['aspects']
        polarities = [asp['polarity'] for asp in aspects]
        pattern = '-'.join(sorted(polarities))
        polarity_patterns[pattern] += 1
        
        # Positional statistics
        if len(aspects) >= 2:
            aspects_sorted = sorted(aspects, key=lambda x: x['from'])
            for i in range(len(aspects_sorted) - 1):
                distance = aspects_sorted[i+1]['from'] - aspects_sorted[i]['to']
                position_stats.append(distance)
    
    print("\n🎯 Polarity Combination Patterns (Top 10):")
    for pattern, count in polarity_patterns.most_common(10):
        percentage = count / len(multi_aspect_sentences) * 100
        print(f"  • {pattern}: {count} occurrences ({percentage:.1f}%)")
    
    print("\n📏 Positional Statistics:")
    if position_stats:
        avg_distance = sum(position_stats) / len(position_stats)
        print(f"  • Average gap: {avg_distance:.2f} tokens")
        print(f"  • Minimum gap: {min(position_stats)} tokens")
        print(f"  • Maximum gap: {max(position_stats)} tokens")
        
        # Distance distribution
        close_count = sum(1 for d in position_stats if d <= 2)
        medium_count = sum(1 for d in position_stats if 3 <= d <= 10)
        far_count = sum(1 for d in position_stats if d > 10)
        
        print(f"  • Adjacent (<=2): {close_count} ({close_count/len(position_stats)*100:.1f}%)")
        print(f"  • Medium-range (3-10): {medium_count} ({medium_count/len(position_stats)*100:.1f}%)")
        print(f"  • Long-range (>10): {far_count} ({far_count/len(position_stats)*100:.1f}%)")
    
    print("\n💡 Main Findings:")
    print("=" * 80)
    mixed_polarity_count = sum(1 for item in multi_aspect_sentences 
                              if len(set(asp['polarity'] for asp in item['aspects'])) > 1)
    print(
        f"1. Polarity diversity: {mixed_polarity_count}/{len(multi_aspect_sentences)} "
        f"({mixed_polarity_count/len(multi_aspect_sentences)*100:.1f}%) of multi-aspect sentences contain different polarities"
    )
    
    positive_dominant = polarity_patterns.get('positive-positive', 0) + polarity_patterns.get('positive-positive-positive', 0)
    print(
        f"2. Positive bias: the positive-positive pattern appears {positive_dominant} times, "
        "suggesting that users often mention multiple positive aspects together"
    )
    
    contrast_in_mixed = sum(1 for sample in selected_samples if 'contrast_words' in sample)
    print(
        f"3. Logical markers: among sentences with polarity contrast, "
        f"{contrast_in_mixed/len([s for s in selected_samples if len(set(s['polarities'])) > 1])*100:.0f}% "
        "contain explicit contrastive conjunctions"
    )
    
    print(
        f"4. Positional distribution: about {close_count/len(position_stats)*100:.0f}% of aspect pairs appear adjacently, "
        "suggesting that related concepts tend to cluster within a sentence"
    )

if __name__ == "__main__":
    generate_final_report()
