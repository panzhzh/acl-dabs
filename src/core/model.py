#!/usr/bin/env python3
# src/core/model.py
# -*- coding: utf-8 -*-

"""
Model loading and configuration module.
Handles tokenizer and model initialization for DeBERTa-based ABSA.
Supports both standard models and advanced dual-layer architecture.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from .. import config
from .dual_layer_model import DualLayerAspectSalienceModel

def load_tokenizer(model_name=config.MODEL_NAME, special_tokens=None):
    """
    Load the tokenizer and optionally append custom special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if special_tokens is not None:
        tokenizer.add_tokens(special_tokens)
    return tokenizer

def _resolve_dtype(dtype_name: str):
    key = (dtype_name or "").strip().lower()
    if key in ("float16", "fp16", "half"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float32", "fp32"):
        return torch.float32
    return torch.bfloat16


def _apply_training_mode(model, mode: str, use_dual_layer: bool, lora_cfg: dict):
    """
    Apply parameter-efficient training mode:
      - full: full fine-tuning (default)
      - freeze_backbone: freeze transformer backbone, train heads/modules only
      - qlora: 4-bit backbone + LoRA adapters
    """
    mode_key = (mode or "full").strip().lower()
    if mode_key in ("full", "full_finetune", "none"):
        return model, {"mode": "full"}

    if mode_key in ("freeze", "freeze_backbone", "head_only", "freeze_head_only"):
        if hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad = False
            # Keep non-backbone modules trainable (DORA/ACBS/classifier).
            for n, p in model.named_parameters():
                if not n.startswith("backbone."):
                    p.requires_grad = True
        else:
            for n, p in model.named_parameters():
                p.requires_grad = any(k in n.lower() for k in ("classifier", "score"))
        return model, {"mode": "freeze_backbone"}

    if mode_key in ("qlora", "lora_4bit", "q_lora"):
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except Exception as e:
            raise RuntimeError("QLoRA requires `peft` to be installed.") from e

        backbone = model.backbone if hasattr(model, "backbone") else model
        backbone = prepare_model_for_kbit_training(backbone)

        target_modules = lora_cfg.get("target_modules")
        if not target_modules:
            model_type = getattr(backbone.config, "model_type", "").lower()
            if "qwen" in model_type or "llama" in model_type or "mistral" in model_type:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
            elif "deberta" in model_type:
                target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
            else:
                target_modules = ["query", "key", "value", "dense"]

        lcfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
        )
        backbone = get_peft_model(backbone, lcfg)
        if hasattr(model, "backbone"):
            model.backbone = backbone
        else:
            model = backbone
        return model, {
            "mode": "qlora",
            "lora_r": int(lora_cfg.get("r", 16)),
            "lora_alpha": int(lora_cfg.get("alpha", 32)),
            "lora_dropout": float(lora_cfg.get("dropout", 0.05)),
            "target_modules": target_modules,
        }

    raise ValueError(f"Unsupported finetune_mode={mode!r}. Use one of: full, freeze_backbone, qlora")


def load_bert_model(
    model_name=config.MODEL_NAME,
    num_labels=3,
    use_dual_layer=False,
    finetune_mode="full",
    qlora_compute_dtype="bfloat16",
    qlora_quant_type="nf4",
    qlora_use_double_quant=True,
    qlora_r=16,
    qlora_alpha=32,
    qlora_dropout=0.05,
    qlora_target_modules=None,
    **ablation_config
):
    """
    Load a sequence classification model while preserving the legacy function name.

    Args:
        model_name: Pretrained model name.
        num_labels: Number of target labels.
        use_dual_layer: Whether to build the dual-layer architecture.
        finetune_mode: One of full / freeze_backbone / qlora.
        **ablation_config: Ablation-study configuration overrides.
    """
    mode_key = (finetune_mode or "full").strip().lower()
    use_qlora = mode_key in ("qlora", "lora_4bit", "q_lora")

    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora_quant_type,
            bnb_4bit_use_double_quant=bool(qlora_use_double_quant),
            bnb_4bit_compute_dtype=_resolve_dtype(qlora_compute_dtype),
        )

    if use_dual_layer:
        # Check ablation configuration and print active components
        active_components = []
        if ablation_config.get('enable_multi_scale', True):
            active_components.append("Multi-Scale Convolution")
        if ablation_config.get('enable_inter_gru', True):
            active_components.append("Inter-Layer GRU")
        if ablation_config.get('enable_aspect_attention', True):
            active_components.append("Aspect-Context Attention")
        if ablation_config.get('enable_context_importance', True):
            active_components.append("Context Importance")
        if ablation_config.get('enable_layer_attention', True):
            active_components.append("Layer-wise Attention")
        
        print("🚀 Loading dual-layer architecture model (Dual-Layer Aspect-Aware Architecture)")
        print(f"   Active components: {', '.join(active_components) if active_components else 'None (baseline configuration)'}")
        
        # Filter out parameters that the model constructor doesn't recognize
        valid_model_params = {
            'enable_multi_scale', 'enable_inter_gru', 'enable_aspect_attention',
            'enable_context_importance', 'enable_layer_attention',
            # Phase B special parameters
            'disable_gating', 'disable_regularization',
            'disable_mask_regularization', 'disable_gate_entropy_reg',
            # Phase C hyperparameters
            'k_value', 'sparse_weight', 'pooling_alternative', 'conv_alternative',
            'context_pooling', 'mask_noise_rho',
            # Depth integration operator (R3: GRU vs LSTM vs non-recurrent)
            'inter_layer_fusion',
            # Layer order ablation
            'layer_order_mode', 'shuffle_seed',
        }
        filtered_config = {k: v for k, v in ablation_config.items() 
                          if k in valid_model_params}

        if use_qlora:
            filtered_config['backbone_quantization_config'] = quantization_config
            filtered_config['backbone_device_map'] = "auto"
            # silence attention fallback warning path for qwen-style backbones
            filtered_config['backbone_attn_implementation'] = "eager"
        else:
            # For very large backbones (e.g. Qwen 7B+), fp32 weights can exceed GPU VRAM
            # once you include optimizer states for the (non-trivial) DABS modules.
            # Allow config-profile to request half-precision loading.
            backbone_dtype = getattr(config, "BACKBONE_TORCH_DTYPE", None)
            if backbone_dtype:
                filtered_config["backbone_torch_dtype"] = backbone_dtype
        
        model = DualLayerAspectSalienceModel(
            model_name=model_name, 
            num_labels=num_labels,
            **filtered_config
        )
    else:
        print("📝 Loading standard backbone model")
        kwargs = dict(
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        backbone_dtype = getattr(config, "BACKBONE_TORCH_DTYPE", None)
        if backbone_dtype and not use_qlora:
            kwargs["torch_dtype"] = backbone_dtype
        if use_qlora:
            kwargs.update(
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation="eager",
            )
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)

    lora_cfg = {
        "r": qlora_r,
        "alpha": qlora_alpha,
        "dropout": qlora_dropout,
        "target_modules": qlora_target_modules,
    }
    model, peft_info = _apply_training_mode(model, finetune_mode, use_dual_layer, lora_cfg)
    print(f"🎯 Fine-tuning mode: {peft_info.get('mode')}")
    if peft_info.get("mode") == "qlora":
        print(
            "   LoRA config: "
            f"r={peft_info['lora_r']}, alpha={peft_info['lora_alpha']}, "
            f"dropout={peft_info['lora_dropout']}, targets={peft_info['target_modules']}"
        )
    
    return model

def move_model_to_device(model):
    """
    Move the model to GPU when available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = model.backbone if hasattr(model, 'backbone') else model
    is_kbit = bool(getattr(backbone, "is_loaded_in_4bit", False) or getattr(backbone, "is_loaded_in_8bit", False))
    if not is_kbit:
        model.to(device)
    else:
        # For k-bit models loaded with device_map, avoid force-moving the full model.
        # But when we have a wrapper model (e.g., dual-layer), move non-backbone modules
        # to the same compute device as backbone.
        if hasattr(model, "backbone"):
            try:
                bb_device = next(model.backbone.parameters()).device
                device = bb_device
            except Exception:
                bb_device = device
            for child_name, child in model.named_children():
                if child_name == "backbone":
                    continue
                child.to(bb_device)
    return model, device
