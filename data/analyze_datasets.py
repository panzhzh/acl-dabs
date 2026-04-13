import json
import pandas as pd
import os

def load_json_dataset(file_path):
    """Load a JSON dataset file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_dataset(data, dataset_name, split_name):
    """Analyze a dataset and count sentiment labels, filtering out empty aspects"""
    stats = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    for item in data:
        # Check if the item has aspects
        aspects = item.get('aspects', [])
        if not aspects:  # Skip if no aspects (empty list)
            continue
            
        # Count sentiment labels for each aspect
        for aspect in aspects:
            if not aspect.get('term'):  # Skip if aspect term is empty
                continue
                
            polarity = aspect.get('polarity', '').lower()
            if polarity == 'positive':
                stats['positive'] += 1
            elif polarity == 'neutral':
                stats['neutral'] += 1  
            elif polarity == 'negative':
                stats['negative'] += 1
    
    print(f"{dataset_name} {split_name}: Positive={stats['positive']}, Neutral={stats['neutral']}, Negative={stats['negative']}")
    return stats

# Dataset file paths mapping
datasets = {
    'Lap14': {
        'train': '/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Laptop_14/Laptops_Train.json',
        'test': '/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Laptop_14/Laptops_Test.json'
    },
    'Res14': {
        'train': '/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_14/Restaurants_Train.json',
        'test': '/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_14/Restaurants_Test.json'
    },
    'Res15': {
        'train': '/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_15/Restaurants_Train.json',
        'test': '/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_15/Restaurants_Test.json'
    },
    'Res16': {
        'train': '/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_16/Restaurants_Train.json',
        'test': '/home/ipanzhzh/pythonCode/paper1_absa/data/semeval/Restaurant_16/Restaurants_Test.json'
    },
    'Twitter': {
        'train': '/home/ipanzhzh/pythonCode/paper1_absa/data/acl-14-short-data-csv/train.json',
        'test': '/home/ipanzhzh/pythonCode/paper1_absa/data/acl-14-short-data-csv/test.json'
    }
}

# Store results
results = {}

# Analyze each dataset
for dataset_name, splits in datasets.items():
    results[dataset_name] = {}
    
    for split_name, file_path in splits.items():
        if os.path.exists(file_path):
            print(f"\nAnalyzing {dataset_name} {split_name}...")
            data = load_json_dataset(file_path)
            results[dataset_name][split_name] = analyze_dataset(data, dataset_name, split_name)
        else:
            print(f"File not found: {file_path}")

print("\n=== FINAL RESULTS ===")
for dataset_name, splits in results.items():
    print(f"{dataset_name}:")
    for split_name, stats in splits.items():
        print(f"  {split_name}: {stats}")

# Generate LaTeX table
def generate_latex_table(results):
    """Generate LaTeX table from results"""
    latex_table = """\\begin{table*}[!ht]
    \\centering
    \\caption{Dataset Statistics.}
    \\label{tab:dataset_statistics_augmented}
    \\begin{tabular}{lccccccc}
        \\toprule
        \\textbf{Dataset} & \\multicolumn{2}{c}{\\textbf{Positive}} & \\multicolumn{2}{c}{\\textbf{Neutral}} & \\multicolumn{2}{c}{\\textbf{Negative}} \\\\
        \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}
        & \\textbf{Train} & \\textbf{Test} & \\textbf{Train} & \\textbf{Test} & \\textbf{Train} & \\textbf{Test} \\\\
        \\midrule
"""
    
    for dataset_name in ['Lap14', 'Res14', 'Res15', 'Res16', 'Twitter']:
        if dataset_name in results:
            train_stats = results[dataset_name].get('train', {'positive': 0, 'neutral': 0, 'negative': 0})
            test_stats = results[dataset_name].get('test', {'positive': 0, 'neutral': 0, 'negative': 0})
            
            latex_table += f"        {dataset_name}     & {train_stats['positive']}   & {test_stats['positive']} & {train_stats['neutral']}   & {test_stats['neutral']} & {train_stats['negative']} & {test_stats['negative']} \\\\\n"
    
    latex_table += """        \\bottomrule
    \\end{tabular}
\\end{table*}"""
    
    return latex_table

# Generate and print LaTeX table
print("\n=== LaTeX TABLE ===")
latex_table = generate_latex_table(results)
print(latex_table)