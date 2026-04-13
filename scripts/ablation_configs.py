#!/usr/bin/env python3
"""
Ablation Study Configuration - Aligned with doc/ablation_set.md

Defines all ablation experiment configurations for systematic analysis:
- Phase A: 4 configs × 4 datasets × 3 seeds = 48 experiments  
- Phase B: 8 LOO configs × 4 datasets × 3 seeds = 96 experiments
- Phase C: 9 configs × 3 datasets × 1 seed = 27 experiments
Total: 207 experiments
"""

# ============================================================================
# Phase A - Main Results (60 experiments)
# ============================================================================
# 4 core configurations representing key architectural contributions

PHASE_A_CONFIGS = {
    'encoder_only': {
        'name': 'Encoder-only',
        'description': 'Only pretrained encoder + classification head (baseline)',
        'use_dual_layer': False,
        'expected_improvement': 0.0,
    },
    
    'dora_only': {
        'name': 'DORA-only', 
        'description': 'DORA enabled (multi-scale conv + inter-GRU), ACBS disabled',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': False,   # ACBS off
        'enable_context_importance': False, # ACBS off (token selection)
        'enable_layer_attention': False,    # ACBS off (layer selection)
        'disable_gating': True,             # Use parameter-free equal fusion
        'disable_regularization': True,     # No regularization
        'expected_improvement': 2.0,
    },
    
    'acbs_only': {
        'name': 'ACBS-only',
        'description': 'DORA disabled, ACBS enabled (aspect-aware selection + fusion)',
        'use_dual_layer': True,
        'enable_multi_scale': False,
        'enable_inter_gru': False,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'disable_regularization': True,  # No regularization for clean attribution
        'expected_improvement': 2.5,
    },
    
    'full_model': {
        'name': 'Full Model',
        'description': 'Complete architecture: DORA + ACBS + regularization',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'disable_regularization': False,
        'sparse_weight': 1e-3,  # Default sparsity regularization (Phase C alignment)
        'expected_improvement': 4.8,
    },
}

# ============================================================================
# Phase B - Leave-One-Out Analysis (120 experiments)
# ============================================================================
# 8 LOO variants: remove one component from full model

PHASE_B_CONFIGS = {
    # B1: Structural LOO (5 variants)
    'full_minus_dora_conv': {
        'name': 'Full - DORA Conv',
        'description': 'Full model with multi-scale convolution disabled',
        'use_dual_layer': True,
        'enable_multi_scale': False,  # Remove this
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'expected_improvement': 3.5,
    },
    
    'full_minus_dora_gru': {
        'name': 'Full - DORA GRU',
        'description': 'Full model with inter-layer GRU disabled',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': False,  # Remove this
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'expected_improvement': 4.0,
    },
    
    'full_minus_token_selection': {
        'name': 'Full - Token Selection',
        'description': 'Full model with token-level selection disabled',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': False,  # Disable token selection (corrected)
        'enable_layer_attention': True,
        'expected_improvement': 3.2,
    },
    
    'full_minus_layer_selection': {
        'name': 'Full - Layer Selection',
        'description': 'Full model with layer-wise selection disabled',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': False,  # Only disable layer selection (corrected)
        'expected_improvement': 3.8,
    },
    
    'full_minus_fusion_gate': {
        'name': 'Full - Fusion Gate',
        'description': 'Full model with gated fusion disabled (equal weights)',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'disable_gating': True,  # Remove gating mechanism
        'expected_improvement': 4.2,
    },
    
    # B2: Regularization LOO (3 variants)
    'full_minus_sparse_reg': {
        'name': 'Full - Sparse Reg',
        'description': 'Full model with sparse regularization disabled (λ_s=0)',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'sparse_weight': 0.0,  # λ_s = 0 (use weight instead of disable flag)
        'expected_improvement': 4.5,
    },
    
    'full_minus_mask_consistency': {
        'name': 'Full - Mask Consistency',
        'description': 'Full model with mask consistency regularization disabled (λ_m=0)',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'disable_mask_regularization': True,  # λ_m = 0 (corrected key)
        'expected_improvement': 4.6,
    },
    
    'full_minus_gate_entropy': {
        'name': 'Full - Gate Entropy',
        'description': 'Full model with gate entropy regularization disabled (λ_ent=0)',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'disable_gate_entropy_reg': True,  # λ_ent = 0
        'expected_improvement': 4.7,
    },
}

# ============================================================================
# Phase C - Engineering & Robustness (27 experiments)
# ============================================================================
# 9 configurations for hyperparameter sensitivity and robustness tests
# Only on 3 datasets: Twitter (1), Laptop14 (2), Restaurant16 (5)

PHASE_C_CONFIGS = {
    # C1: Structural & Hyperparameter Sensitivity (8 variants)
    'k_sensitivity_2': {
        'name': 'K=2 Sensitivity',
        'description': 'Full model with K=2 (fewer recent layers)',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'k_value': 2,
        'sparse_weight': 0.001,  # Should use default regularization strength
        'expected_improvement': 4.5,
    },
    
    'k_sensitivity_4': {
        'name': 'K=4 Sensitivity', 
        'description': 'Full model with K=4 (default)',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'k_value': 4,
        'sparse_weight': 0.001,  # Should use default regularization strength
        'expected_improvement': 4.8,
    },
    
    'k_sensitivity_6': {
        'name': 'K=6 Sensitivity',
        'description': 'Full model with K=6 (more recent layers)', 
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'k_value': 6,
        'sparse_weight': 0.001,  # Should use default regularization strength
        'expected_improvement': 4.7,
    },

    'k_sensitivity_8': {
        'name': 'K=8 Sensitivity',
        'description': 'Full model with K=8 (deeper aggregation for trade-off study)',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'k_value': 8,
        'sparse_weight': 0.001,
        'expected_improvement': 4.6,
    },
    
    'sparse_weight_0': {
        'name': 'λ_s=0 Sparsity',
        'description': 'Full model with no sparse regularization',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'sparse_weight': 0.0,
        'expected_improvement': 4.5,
    },
    
    'sparse_weight_1e3': {
        'name': 'λ_s=1e-3 Sparsity',
        'description': 'Full model with default sparse regularization',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'sparse_weight': 1e-3,
        'expected_improvement': 4.8,
    },
    
    'sparse_weight_5e3': {
        'name': 'λ_s=5e-3 Sparsity',
        'description': 'Full model with higher sparse regularization',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'sparse_weight': 5e-3,
        'expected_improvement': 4.6,
    },
    
    'pooling_cls': {
        'name': 'CLS Pooling',
        'description': 'Full model with CLS pooling instead of mean pooling',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'context_pooling': 'cls',
        'sparse_weight': 0.001,  # Should use default regularization strength
        'expected_improvement': 4.7,
    },
    
    'conv_standard': {
        'name': 'Standard Conv',
        'description': 'Full model with standard convolution instead of depthwise',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'conv_alternative': True,
        'sparse_weight': 0.001,  # Should use default regularization strength
        'expected_improvement': 4.8,
    },
    
    # C2: Robustness (1 variant)
    'mask_noise_03': {
        'name': 'Mask Noise ρ=0.3',
        'description': 'Full model with 0.3 probability mask noise during training',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'mask_noise_rho': 0.3,
        'sparse_weight': 0.001,  # Should use default regularization strength
        'expected_improvement': 4.4,
    },
}

# Optional extra configs for Phase A figures (not included in main Phase A table)
EXTRA_A_CONFIGS = {
    'dora_acbs_no_reg': {
        'name': 'DORA+ACBS (no-reg)',
        'description': 'DORA and ACBS both enabled without regularization',
        'use_dual_layer': True,
        'enable_multi_scale': True,
        'enable_inter_gru': True,
        'enable_aspect_attention': True,
        'enable_context_importance': True,
        'enable_layer_attention': True,
        'disable_gating': False,
        'disable_regularization': True,
        'expected_improvement': 4.6,
    }
}

# ============================================================================
# Phase Experiment Plans
# ============================================================================

DATASETS = {
    2: 'Laptop-14', 
    3: 'Restaurant-14',
    4: 'Restaurant-15',
    5: 'Restaurant-16'
}

MODELS = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'deberta': 'microsoft/deberta-v3-base'
}

# Phase-specific dataset selections
ALL_DATASETS = ['Laptop-14', 'Restaurant-14', 'Restaurant-15', 'Restaurant-16']
PHASE_C_DATASETS = ['Laptop-14', 'Restaurant-15', 'Restaurant-16']  # Only 3 datasets for Phase C

PHASE_EXPERIMENT_PLANS = {
    'phase_A': {
        'name': 'Phase A - Main Results',
        'description': 'Core architectural contributions across 4 datasets',
        'configs': PHASE_A_CONFIGS,
        'datasets': ALL_DATASETS,
        'models': list(MODELS.keys()),  # Use all models for Phase A
        'seeds': [42, 123,234,345, 456, 12,23,34,1234,2345,420,421,422],
        'total_experiments': 48,  # 4 configs × 4 datasets × 3 seeds × 3 models
    },
    
    'phase_B': {
        'name': 'Phase B - Leave-One-Out Analysis',
        'description': 'Component necessity analysis across 4 datasets',
        'configs': PHASE_B_CONFIGS,
        'datasets': ALL_DATASETS,
        'models': ['deberta'],  # Only DeBERTa for Phase B
        'seeds': [42, 123, 456],
        'total_experiments': 96,  # 8 configs × 4 datasets × 3 seeds
    },
    
    'phase_C': {
        'name': 'Phase C - Engineering & Robustness',
        'description': 'Hyperparameter sensitivity and robustness on 3 representative datasets',
        'configs': PHASE_C_CONFIGS,
        'datasets': PHASE_C_DATASETS,
        'models': ['deberta'],  # Only DeBERTa for Phase C
        'seeds': [42],  # Single seed for efficiency
        'total_experiments': 27,  # 9 configs × 3 datasets × 1 seed
    }
}

# ============================================================================
# Utility Functions
# ============================================================================

def get_config(config_name):
    """Get configuration by name from any phase"""
    all_configs = {}
    all_configs.update(PHASE_A_CONFIGS)
    all_configs.update(PHASE_B_CONFIGS) 
    all_configs.update(PHASE_C_CONFIGS)
    
    if config_name not in all_configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(all_configs.keys())}")
    return all_configs[config_name].copy()

def get_phase_config(phase_name):
    """Get phase configuration by name"""
    if phase_name not in PHASE_EXPERIMENT_PLANS:
        raise ValueError(f"Unknown phase: {phase_name}. Available: {list(PHASE_EXPERIMENT_PLANS.keys())}")
    return PHASE_EXPERIMENT_PLANS[phase_name]

def get_phase_configs_dict(phase_name):
    """Get all configs for a phase as a unified dictionary"""
    if phase_name == 'phase_A':
        return PHASE_A_CONFIGS
    elif phase_name == 'phase_B':
        return PHASE_B_CONFIGS
    elif phase_name == 'phase_C':
        return PHASE_C_CONFIGS
    else:
        raise ValueError(f"Unknown phase: {phase_name}")

def list_phase_experiments(phase_name):
    """List all experiments for a phase"""
    phase_config = get_phase_config(phase_name)
    experiments = []
    
    # Get models for this phase (default to deberta if not specified)
    models = phase_config.get('models', ['deberta'])
    
    for model in models:
        for dataset in phase_config['datasets']:
            for config_name in phase_config['configs'].keys():
                for seed in phase_config['seeds']:
                    experiments.append({
                        'phase': phase_name,
                        'model': model,
                        'config_name': config_name,
                        'dataset': dataset,
                        'seed': seed
                    })
    
    return experiments

def print_experiment_summary():
    """Print summary of all experiments"""
    total_experiments = 0
    
    print("🔬 ABLATION STUDY EXPERIMENT PLAN")
    print("=" * 50)
    
    for phase_name, phase_config in PHASE_EXPERIMENT_PLANS.items():
        print(f"\n📊 {phase_config['name']}")
        print(f"   Description: {phase_config['description']}")
        print(f"   Configs: {len(phase_config['configs'])}")
        print(f"   Datasets: {len(phase_config['datasets'])} {phase_config['datasets']}")
        print(f"   Seeds: {phase_config['seeds']}")
        print(f"   Total: {phase_config['total_experiments']} experiments")
        total_experiments += phase_config['total_experiments']
    
    print(f"\n🎯 GRAND TOTAL: {total_experiments} experiments")
    print(f"   Phase A: {PHASE_EXPERIMENT_PLANS['phase_A']['total_experiments']} experiments")
    print(f"   Phase B: {PHASE_EXPERIMENT_PLANS['phase_B']['total_experiments']} experiments") 
    print(f"   Phase C: {PHASE_EXPERIMENT_PLANS['phase_C']['total_experiments']} experiments")

if __name__ == "__main__":
    print_experiment_summary()
