#!/usr/bin/env python3
"""
Automated Multilingual Ablation Study for DABS Architecture.

This script runs a comprehensive ablation study on three multilingual datasets
(French, Russian, Spanish) using the microsoft/mdeberta-v3-base model.

It tests four configurations:
1. Baseline (Encoder-only)
2. DORA-only
3. ACBS-only
4. Full Model (DORA + ACBS)

Each configuration is run with 3 different random seeds (42, 123, 456).

The script generates a single JSON file with all results and saves the best
checkpoint for each language based on Macro-F1 score.

Usage:
    python scripts/run_multilingual_ablation.py
"""

import os
import sys
import subprocess
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import argparse

# Add project root to path to import ablation_configs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from scripts.ablation_configs import PHASE_A_CONFIGS

# --- Configuration ---

# 1. Model to use
MODEL_NAME = "microsoft/mdeberta-v3-base"

# 2. Datasets to test
# The key is the language code, the value is the DATASET_CHOICE ID
# that will be passed to train.py.
# Current run target: Dutch (DU) + Turkish (TU).
# If you want to compare with Russian as well, add: "RU": 7
MULTILINGUAL_DATASETS = {
    "FR": 6,
    "RU": 7,
    "ES": 8,
    "DU": 9,
    "TU": 10,
}

# 3. Ablation configurations to run
CONFIGS_TO_RUN = {
    "baseline": PHASE_A_CONFIGS['encoder_only'],
    "dora_only": PHASE_A_CONFIGS['dora_only'],
    "acbs_only": PHASE_A_CONFIGS['acbs_only'],
    "full_model": PHASE_A_CONFIGS['full_model'],
}

# 4. Seeds for reproducibility
SEEDS = [42,123,456]

# --- Logger and Runner ---

class AblationLogger:
    """Simple logger for the ablation study."""
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.log_file = self.results_dir / "multilingual_ablation_log.txt"

    def log(self, message):
        """Log message to console and file with a timestamp."""
        timestamp = datetime.now(timezone.utc).strftime('[%Y-%m-%d %H:%M:%S UTC]')
        log_entry = f"{timestamp} {message}"
        print(log_entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

def run_experiment(exp_config, logger, use_test_mode=False):
    """Runs a single training experiment and returns the results."""
    lang = exp_config['lang']
    config_name = exp_config['config_name']
    seed = exp_config['seed']
    dataset_id = exp_config['dataset_id']
    model_config = exp_config['model_config']

    logger.log(f"🚀 Starting Experiment: Lang={lang}, Config={config_name}, Seed={seed}")

    cmd = ["python", "scripts/train.py"]
    if model_config.get('use_dual_layer', False):
        cmd.append("--dual-layer")

    env = os.environ.copy()
    env['DATASET_CHOICE'] = str(dataset_id)
    env['MODEL_NAME'] = MODEL_NAME
    env['RANDOM_SEED'] = str(seed)
    env['PYTHONHASHSEED'] = str(seed)
    env['LANG_CODE'] = str(lang)
    env['CONFIG_NAME'] = str(config_name)
    
    if use_test_mode:
        env['USE_TEST_MODE'] = '1'

    ablation_params = {k: v for k, v in model_config.items() if k != 'name' and k != 'description'}
    env['ABLATION_CONFIG'] = json.dumps(ablation_params)

    logger.log(f"   Running command: {' '.join(cmd)}")
    logger.log(f"   With ENV: DATASET_CHOICE={dataset_id}, RANDOM_SEED={seed}")
    if use_test_mode:
        logger.log("   [TEST MODE] Using 1% of the dataset.")
    logger.log(f"   ABLATION_CONFIG: {env['ABLATION_CONFIG']}")

    try:
        process = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True, # Will raise CalledProcessError if train.py fails
            cwd=PROJECT_ROOT
        )

        # Find the JSON output line from the training script
        for line in process.stdout.splitlines():
            if line.startswith("ABLATION_RESULTS_JSON:"):
                result_json_str = line.replace("ABLATION_RESULTS_JSON:", "").strip()
                results = json.loads(result_json_str)
                logger.log(f"✅ Experiment SUCCEEDED. MF1: {results.get('macro_f1', 'N/A'):.4f}, Acc: {results.get('accuracy', 'N/A'):.4f}")
                
                # Add experiment metadata to the results
                results['experiment_details'] = exp_config
                
                # Find the path of the saved checkpoint
                for log_line in process.stdout.splitlines():
                    if "Best model saved to" in log_line:
                        checkpoint_path = log_line.split("Best model saved to")[-1].strip()
                        results['checkpoint_path'] = checkpoint_path
                        break
                
                return results

    except subprocess.CalledProcessError as e:
        logger.log(f"❌ Experiment FAILED: Lang={lang}, Config={config_name}, Seed={seed}")
        logger.log(f"   Return Code: {e.returncode}")
        logger.log(f"   STDOUT: {e.stdout}")
        logger.log(f"   STDERR: {e.stderr}")
        return None
    except Exception as e:
        logger.log(f"💥 An unexpected error occurred: {e}")
        return None

def main():
    """Main function to orchestrate the multilingual ablation study."""
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Run multilingual ablation study for DABS.")
    parser.add_argument('--test', action='store_true', help='Run in test mode using 1%% of the dataset.')
    args = parser.parse_args()
    # Create a timestamped directory for all results of this run
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    results_dir = Path(PROJECT_ROOT) / "results" / f"multilingual_ablation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = AblationLogger(results_dir)
    logger.log("===== Starting Multilingual Ablation Study =====")
    logger.log(f"Model: {MODEL_NAME}")
    logger.log(f"Languages: {list(MULTILINGUAL_DATASETS.keys())}")
    logger.log(f"Configurations: {list(CONFIGS_TO_RUN.keys())}")
    logger.log(f"Seeds: {SEEDS}")
    total_runs = len(MULTILINGUAL_DATASETS) * len(CONFIGS_TO_RUN) * len(SEEDS)
    logger.log(f"Total experiments to run: {total_runs}")
    if args.test:
        logger.log("⚡️ TEST MODE IS ENABLED. All runs will use a 1% data sample.")
    logger.log("==================================================")

    all_results = []
    best_checkpoints = {} # {lang: {'mf1': score, 'path': path, 'config': config_name}}

    run_count = 0
    for lang, dataset_id in MULTILINGUAL_DATASETS.items():
        best_checkpoints[lang] = {'mf1': -1, 'path': None, 'config': None}
        for config_name, model_config in CONFIGS_TO_RUN.items():
            for seed in SEEDS:
                run_count += 1
                logger.log(f"\n--- Running experiment {run_count}/{total_runs} ---")
                
                exp_config = {
                    'lang': lang,
                    'dataset_id': dataset_id,
                    'config_name': config_name,
                    'seed': seed,
                    'model_config': model_config
                }

                result = run_experiment(exp_config, logger, use_test_mode=args.test)

                if result:
                    all_results.append(result)

                    # Check if this is the best model for the current language
                    current_mf1 = result.get('macro_f1', -1)
                    if current_mf1 > best_checkpoints[lang]['mf1']:
                        best_checkpoints[lang]['mf1'] = current_mf1
                        best_checkpoints[lang]['path'] = result.get('checkpoint_path')
                        best_checkpoints[lang]['config'] = config_name
                        best_checkpoints[lang]['seed'] = seed

                    # Save one JSON file per language/config/seed, including epoch_logs.
                    single_result_file = results_dir / f"{lang}_{config_name}_seed{seed}.json"
                    with open(single_result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    logger.log(f"   Saved detailed results to: {single_result_file}")

    # --- Aggregate results by language and save a separate JSON file for each one. ---
    for lang in MULTILINGUAL_DATASETS.keys():
        lang_results = [r for r in all_results if r.get('experiment_details', {}).get('lang') == lang]
        lang_json_path = results_dir / f"{lang}_all_configs.json"

        lang_data = {
            'language': lang,
            'model': MODEL_NAME,
            'configs': list(CONFIGS_TO_RUN.keys()),
            'seeds': SEEDS,
            'timestamp': timestamp,
            'results': lang_results
        }

        with open(lang_json_path, 'w', encoding='utf-8') as f:
            json.dump(lang_data, f, indent=2)
        logger.log(f"   Saved {lang} aggregated results to: {lang_json_path}")

    # --- Save all results to a single JSON file ---
    final_results_path = results_dir / "multilingual_ablation_summary.json"
    summary_data = {
        'study_parameters': {
            'model': MODEL_NAME,
            'languages': list(MULTILINGUAL_DATASETS.keys()),
            'configs': list(CONFIGS_TO_RUN.keys()),
            'seeds': SEEDS,
            'timestamp': timestamp
        },
        'best_checkpoints_info': best_checkpoints,
        'all_run_results': all_results
    }
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    logger.log(f"\n\n✅ Full ablation study complete. Summary saved to: {final_results_path}")

    # --- Copy best checkpoints to a dedicated folder ---
    best_checkpoints_dir = results_dir / "best_checkpoints"
    best_checkpoints_dir.mkdir(exist_ok=True)
    
    logger.log("\n--- Saving Best Checkpoints ---")
    for lang, best_info in best_checkpoints.items():
        if best_info['path'] and Path(best_info['path']).exists():
            src_path = Path(best_info['path'])
            dest_name = f"best_model_{lang}_{best_info['config']}_seed{best_info['seed']}"
            dest_path = best_checkpoints_dir / dest_name
            
            shutil.copytree(src_path, dest_path)
            logger.log(f"   Copied best model for {lang} to: {dest_path}")
        else:
            logger.log(f"   ⚠️ Could not find best model for {lang}. Path: {best_info['path']}")

    logger.log("\n===== Multilingual Ablation Study Finished =====")

if __name__ == "__main__":
    main()
