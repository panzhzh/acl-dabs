#!/usr/bin/env python3
"""
DORA+ACBS Architecture for Aspect-Based Sentiment Analysis

Paper: "Hierarchical Multi-Scale Aspect-Aware Transformer for Fine-Grained Sentiment Analysis: 
       A Dual-Layer Architecture Approach"

Authors: [Your Name]
Date: 2025-01-04

Architecture (paper Section 3.2):
- DORA (Deep Ordered Recursive Aggregation): Context-Agnostic Multi-Scale Feature Extraction
  - Inter-layer recursive aggregation: GRU-based information passing across layers
  - Multi-scale convolution enhancement: word/phrase/syntax-level modeling
  
- ACBS (Aspect-Conditioned Bilateral Selection): Aspect-Aware Contextual Focusing  
  - Aspect-context interaction: semantic interaction learning between aspect and sentence
  - Sequence-dimension selection: dynamic context importance weighting (token-level selection)
  - Layer-dimension selection: adaptive layer-wise attention (layer-level selection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from .. import config


def _resolve_torch_dtype(dtype_value):
    """
    Resolve user-provided dtype string/object to a torch dtype.
    Returns one of torch.dtype / "auto" / None.
    """
    if dtype_value is None:
        return None
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if isinstance(dtype_value, str):
        key = dtype_value.strip().lower()
        if key in ("auto",):
            return "auto"
        if key in ("float16", "fp16", "half"):
            return torch.float16
        if key in ("bfloat16", "bf16"):
            return torch.bfloat16
        if key in ("float32", "fp32"):
            return torch.float32
    raise ValueError(
        f"Unsupported backbone_torch_dtype={dtype_value!r}. "
        "Use one of: auto, float16/fp16, bfloat16/bf16, float32/fp32."
    )


class DualLayerAspectSalienceModel(nn.Module):
    """
    DORA+ACBS Architecture for ABSA (paper Algorithm 1):
    
    DORA (Deep Ordered Recursive Aggregation): shared modeling stage
    - Multi-scale sequence shaping: improves general semantic understanding across granularities
    - Inter-layer information flow: inter-layer GRU for depth-wise information passing
    
    ACBS (Aspect-Conditioned Bilateral Selection): aspect-conditioned selection stage
    - Sequence-dimension selection: token-level aspect-conditioned selection (w_t)
    - Layer-dimension selection: layer-level aspect-conditioned selection (alpha_k)
    - Multi-source fusion: three-source fusion for the final representation
    """
    
    def __init__(self, model_name=config.MODEL_NAME, num_labels=3,
                 # Backbone loading options
                 backbone_torch_dtype=None,
                 backbone_low_cpu_mem_usage=True,
                 backbone_quantization_config=None,
                 backbone_device_map=None,
                 backbone_attn_implementation=None,
                 # Ablation study switches
                 enable_multi_scale=False,
                 enable_inter_gru=False,
                 inter_layer_fusion: str = 'gru',
                 enable_aspect_attention=False,
                 enable_context_importance=False,
                 enable_layer_attention=False,
                 # Phase B special parameters
                 disable_gating=False,
                 disable_regularization=False,
                 disable_mask_regularization=False,
                 disable_gate_entropy_reg=False,
                 # Phase C hyperparameters
                 k_value=None,
                 sparse_weight=None,
                 pooling_alternative=False,
                 conv_alternative=False,
                 context_pooling: str = 'mean',
                 mask_noise_rho: float = 0.0,
                 # Layer order ablation
                 layer_order_mode: str = 'normal',
                 shuffle_seed: int = 42):
        super().__init__()

        # Base transformer model (requires hidden states for multi-layer analysis).
        backbone_kwargs = {
            "num_labels": num_labels,
            "output_hidden_states": True,  # Essential for multi-layer processing
            # NOTE: For some backbones (e.g. Qwen2.*) enabling `output_attentions=True`
            # under fp16 autocast can produce NaNs in hidden states/logits. We keep it
            # off by default because the architecture does not require attention maps.
            "output_attentions": False,
        }
        resolved_dtype = _resolve_torch_dtype(backbone_torch_dtype)
        if resolved_dtype is not None:
            backbone_kwargs["torch_dtype"] = resolved_dtype
        if backbone_low_cpu_mem_usage is not None:
            backbone_kwargs["low_cpu_mem_usage"] = bool(backbone_low_cpu_mem_usage)
        if backbone_quantization_config is not None:
            backbone_kwargs["quantization_config"] = backbone_quantization_config
        if backbone_device_map is not None:
            backbone_kwargs["device_map"] = backbone_device_map
        if backbone_attn_implementation is not None:
            backbone_kwargs["attn_implementation"] = backbone_attn_implementation

        try:
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                **backbone_kwargs
            )
        except TypeError:
            # Backward compatibility for older transformers versions.
            backbone_kwargs.pop("low_cpu_mem_usage", None)
            backbone_kwargs.pop("attn_implementation", None)
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                **backbone_kwargs
            )
        
        self.hidden_size = self.backbone.config.hidden_size
        self.num_labels = num_labels
        self.num_layers = self.backbone.config.num_hidden_layers
        
        # Store ablation configuration
        self.enable_multi_scale = enable_multi_scale
        self.enable_inter_gru = enable_inter_gru
        self.inter_layer_fusion = (inter_layer_fusion or 'gru').strip().lower()
        self.enable_aspect_attention = enable_aspect_attention
        self.enable_context_importance = enable_context_importance
        self.enable_layer_attention = enable_layer_attention
        
        # Store Phase B special parameters
        self.disable_gating = disable_gating
        self.disable_regularization = disable_regularization
        self.disable_mask_regularization = disable_mask_regularization
        self.disable_gate_entropy_reg = disable_gate_entropy_reg
        
        # Store Phase C hyperparameters
        self.k_value = k_value if k_value is not None else 6  # Default K=6 for layer selection
        self.sparse_weight = sparse_weight if sparse_weight is not None else 0.0  # Default no sparsity
        self.pooling_alternative = pooling_alternative
        self.conv_alternative = conv_alternative
        self.context_pooling = context_pooling if context_pooling in ('mean', 'cls') else 'mean'
        self.mask_noise_rho = float(mask_noise_rho) if mask_noise_rho is not None else 0.0
        self.mask_consistency_weight = 1e-3  # default λ_m

        # Layer order ablation parameters
        self.layer_order_mode = layer_order_mode
        self.shuffle_seed = shuffle_seed
        # Permutation computed lazily in forward to match actual K
        
        # ===== DORA: shared modeling (Deep Ordered Recursive Aggregation) =====
        
        # Inter-layer recursive aggregation for information flow across transformer layers (paper Section 3.3.2)
        self.dora_inter_gru_cell = None
        self.dora_inter_lstm_cell = None
        if self.enable_inter_gru:
            if self.inter_layer_fusion == 'gru':
                # Implement the recurrence with GRUCell: s_{u,t} = GRUCell(H^{(L-K+u)}_t, s_{u-1,t})
                self.dora_inter_gru_cell = nn.GRUCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size
                )
            elif self.inter_layer_fusion == 'lstm':
                # LSTM alternative (R3 request): compare against GRU for ordered depth integration.
                self.dora_inter_lstm_cell = nn.LSTMCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size
                )
            elif self.inter_layer_fusion in ('cumavg', 'cumulative_avg', 'mean'):
                # Non-recurrent baseline: cumulative mean of previous layers (ordered, parameter-free).
                pass
            else:
                raise ValueError(
                    f"Invalid inter_layer_fusion={self.inter_layer_fusion!r}. "
                    f"Supported: 'gru', 'lstm', 'cumavg'."
                )
        
        # Multi-scale convolution enhancement for different semantic granularities (paper Section 3.3.1)
        if self.enable_multi_scale:
            self.dora_multi_scale_convs = nn.ModuleList([])
            # Allocate channels so concatenated multi-scale features exactly match hidden_size
            base = self.hidden_size // 3
            channel_splits = [base, base, self.hidden_size - 2 * base]
            for k, out_channels in zip([1, 3, 5], channel_splits):  # Multi-scale semantic granularities: word, phrase, syntax
                if self.conv_alternative:
                    # Standard convolution alternative (for Phase C comparison)
                    standard_conv = nn.Conv1d(
                        in_channels=self.hidden_size,
                        out_channels=out_channels,
                        kernel_size=k,
                        padding=k//2
                    )
                    self.dora_multi_scale_convs.append(standard_conv)
                else:
                    # Use depthwise separable convolution by default.
                    depthwise_separable = nn.Sequential(
                        # Depthwise: convolve each channel independently.
                        nn.Conv1d(
                            in_channels=self.hidden_size,
                            out_channels=self.hidden_size,
                            kernel_size=k,
                            padding=k//2,
                            groups=self.hidden_size  # Depthwise
                        ),
                        # Pointwise: mix channels with a 1x1 convolution.
                        nn.Conv1d(
                            in_channels=self.hidden_size,
                            out_channels=out_channels,
                            kernel_size=1
                        )
                    )
                    self.dora_multi_scale_convs.append(depthwise_separable)
            
            # Multi-scale feature fusion for the convolution outputs.
            self.dora_scale_fusion = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.dora_multi_scale_convs = None
            self.dora_scale_fusion = None
        
        # ===== ACBS: aspect-conditioned selection (Aspect-Conditioned Bilateral Selection) =====
        
        # Aspect-context interaction learning (paper Section 3.4.1)
        if self.enable_aspect_attention:
            # Use model's native num_attention_heads or default to 12
            num_heads = getattr(self.backbone.config, 'num_attention_heads', 12)
            # Ensure num_heads divides hidden_size evenly
            if self.hidden_size % num_heads != 0:
                num_heads = 12  # Fallback to 12 if not divisible
            
            self.acbs_aspect_context_mha = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.acbs_aspect_context_mha = None
        
        # Sequence-dimension selection: dynamic context importance learning (paper Section 3.4.2)
        if self.enable_context_importance:
            # Independent sigmoid gating: w_t = sigma(MLP([C_t; a])) (paper Eq. 163)
            self.acbs_token_gating_net = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),  # concat aspect + context
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()  # Independent gating supports parallel evidence.
            )
        else:
            self.acbs_token_gating_net = None
        
        # Layer-dimension selection: adaptive layer-wise attention (paper Section 3.4.3)
        if self.enable_layer_attention:
            # Input: [aspect_repr; gate-weighted local_ctx] -> hidden_size * 2
            self.acbs_layer_attention = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size // 2),  # aspect + local context
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, self.num_layers),  # Attention over all layers
                nn.Softmax(dim=-1)
            )
            self.attention_layers = self.num_layers
        else:
            self.acbs_layer_attention = None
        
        # Multi-granularity aspect representation learning.
        self.aspect_multi_granularity = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(3)  # word-level, phrase-level, semantic-level
        ])
        
        # Three-source fusion: gated fusion vs. direct projection (Phase B ablation).
        if self.disable_gating:
            # Parameter-free equal weighting (no extra params)
            self.acbs_no_gate_projection = None
            self.acbs_fusion_gate = None
        else:
            # Three-source gated fusion (paper Eqs. 197-198)
            self.acbs_fusion_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),  # [ĉc;ĉl;ĉa]
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, 3),  # [g1, g2, g3]
                nn.Softmax(dim=-1)
            )
            self.acbs_no_gate_projection = None
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, num_labels)
        )

    @staticmethod
    def _filter_forward_kwargs(kwargs):
        """Drop Trainer-specific keys unsupported by the backbone/model forward."""
        return {
            k: v for k, v in kwargs.items()
            if k not in ['num_items_in_batch', 'pos_features']
        }

    def _prepare_aspect_inputs(self, sequence_features, aspect_token_positions=None, aspect_mask=None):
        """Normalize optional aspect inputs for both standard and reuse inference paths."""
        batch_size, seq_len = sequence_features.shape[:2]
        device = sequence_features.device
        if aspect_token_positions is None:
            aspect_token_positions = [(0, 1)] * batch_size
        if aspect_mask is None:
            aspect_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
        return aspect_token_positions, aspect_mask

    def encode_shared(self, input_ids, attention_mask=None, **kwargs):
        """
        Run the shared sentence computation once: backbone + DORA + context reorganization.
        """
        filtered_kwargs = self._filter_forward_kwargs(kwargs)
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **filtered_kwargs
        )

        all_hidden_states = backbone_outputs.hidden_states
        last_hidden = all_hidden_states[-1]
        layers_to_use = min(self.k_value, len(all_hidden_states) - 1)
        selected_layers = all_hidden_states[-layers_to_use:]

        if self.enable_multi_scale:
            enhanced_features = self.dora_multi_scale_sequence_shaping(last_hidden)
        else:
            enhanced_features = last_hidden

        if self.enable_inter_gru:
            layer_enhanced_features = self.dora_cross_layer_information_flow(selected_layers)
        else:
            layer_enhanced_features = selected_layers

        if self.enable_aspect_attention:
            context_enhanced, interaction_weights = self.acbs_aspect_context_mha(
                query=enhanced_features,
                key=enhanced_features,
                value=enhanced_features,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
        else:
            context_enhanced = enhanced_features
            interaction_weights = None

        return {
            'backbone_outputs': backbone_outputs,
            'all_hidden_states': all_hidden_states,
            'enhanced_features': enhanced_features,
            'context_enhanced': context_enhanced,
            'interaction_weights': interaction_weights,
            'layer_enhanced_features': layer_enhanced_features,
            'layers_to_use': layers_to_use,
            'attention_mask': attention_mask,
        }

    def forward_aspect_from_shared(
        self,
        shared_output,
        aspect_token_positions=None,
        aspect_mask=None,
        labels=None,
        return_intermediates=False,
    ):
        """
        Run the aspect-specific ACBS readout on top of precomputed shared sentence features.
        """
        context_enhanced = shared_output['context_enhanced']
        enhanced_features = shared_output['enhanced_features']
        layer_enhanced_features = shared_output['layer_enhanced_features']
        layers_to_use = shared_output['layers_to_use']
        attention_mask = shared_output.get('attention_mask')
        backbone_outputs = shared_output['backbone_outputs']
        all_hidden_states = shared_output['all_hidden_states']

        batch_size, seq_len, _ = context_enhanced.shape
        device = context_enhanced.device
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

        aspect_token_positions, aspect_mask = self._prepare_aspect_inputs(
            context_enhanced,
            aspect_token_positions=aspect_token_positions,
            aspect_mask=aspect_mask,
        )

        if self.training and aspect_mask is not None and self.mask_noise_rho > 0:
            noise = (torch.rand_like(aspect_mask.float()) < self.mask_noise_rho).float()
            aspect_mask = (aspect_mask.float() * (1 - noise) + (1 - aspect_mask.float()) * noise).clamp(0, 1)
            aspect_mask = aspect_mask.long()

        need_aspect = (
            self.enable_aspect_attention or self.enable_context_importance or self.enable_layer_attention
        )
        aspect_repr = None
        if need_aspect:
            aspect_repr = self.acbs_aspect_representation(
                enhanced_features, aspect_token_positions, aspect_mask
            )

        if self.enable_context_importance:
            context_importance = self.acbs_sequence_dimension_selection(
                context_enhanced, aspect_repr
            )
        else:
            context_importance = torch.ones_like(context_enhanced[:, :, :1])

        if attention_mask is not None:
            context_importance = context_importance * attention_mask.unsqueeze(-1).to(context_importance.dtype)

        adaptive_layer_weights = None
        if self.enable_layer_attention:
            pooled_context = self._masked_sequence_mean(context_enhanced, attention_mask)
            aspect_context = torch.cat([aspect_repr, pooled_context], dim=-1)
            weights_wide = self.acbs_layer_attention(aspect_context)
            adaptive_layer_weights = weights_wide[:, -layers_to_use:]
            denom = adaptive_layer_weights.sum(dim=-1, keepdim=True).float().clamp_min(1e-6)
            adaptive_layer_weights = (adaptive_layer_weights.float() / denom).to(dtype=weights_wide.dtype)
            weighted_layer_features = self.acbs_layer_dimension_selection(
                layer_enhanced_features, adaptive_layer_weights
            )
        else:
            if isinstance(layer_enhanced_features, list):
                stacked_features = torch.stack(layer_enhanced_features, dim=2)
                weighted_layer_features = stacked_features.mean(dim=2)
            else:
                weighted_layer_features = layer_enhanced_features

        final_representation, fusion_components = self.acbs_multi_source_fusion(
            context_enhanced,
            weighted_layer_features,
            context_importance,
            aspect_repr,
            attention_mask=attention_mask,
        )
        logits = self.classifier(final_representation)

        total_loss = None
        if labels is not None:
            focal_loss = self._compute_focal_loss(logits, labels)
            reg_sparse = 0.0
            reg_gate = 0.0
            reg_mask = 0.0

            if not self.disable_regularization:
                reg_sparse = self._compute_sparsity_regularization(context_importance, attention_mask)
                if not self.disable_gate_entropy_reg and not self.disable_gating and 'gate_weights' in fusion_components:
                    reg_gate = self._compute_gate_entropy_regularization(fusion_components['gate_weights'])
                if not self.disable_mask_regularization and aspect_mask is not None and self.enable_context_importance:
                    m = aspect_mask.float()
                    denom = m.sum().float().clamp_min(1.0)
                    gates = context_importance.squeeze(-1).float()
                    reg_mask = (gates * m).sum() / denom

            total_loss = (
                focal_loss
                + self.sparse_weight * reg_sparse
                + 1e-2 * reg_gate
                + self.mask_consistency_weight * reg_mask
            )

        result = {
            'loss': total_loss,
            'logits': logits
        }

        if return_intermediates:
            probs = F.softmax(logits, dim=-1)
            result.update({
                'final_representation': final_representation,
                'layer_weights': adaptive_layer_weights,
                'gate_weights': fusion_components['gate_weights'] if fusion_components and 'gate_weights' in fusion_components else None,
                'context_importance': context_importance,
                'aspect_representation': aspect_repr,
                'context_representation': fusion_components['c_hat'] if fusion_components and 'c_hat' in fusion_components else None,
                'layer_representation': fusion_components['l_hat'] if fusion_components and 'l_hat' in fusion_components else None,
                'probabilities': probs,
                'hidden_states': all_hidden_states,
            })

        if self.training and fusion_components is not None:
            if 'gate_weights' in fusion_components:
                result['gate_statistics'] = self.compute_gate_statistics(fusion_components['gate_weights'])
            if context_importance is not None:
                result['token_statistics'] = self.compute_token_statistics(context_importance, aspect_mask)

        if self.training:
            result.update({
                'hidden_states': all_hidden_states,
                'attentions': backbone_outputs.attentions
            })

        return result
        
    def forward(self, input_ids, attention_mask=None, aspect_token_positions=None, 
                aspect_mask=None, labels=None, return_intermediates=False, **kwargs):
        """
        Forward pass through the DORA+ACBS architecture (paper Algorithm 1)
        
        Args:
            input_ids: [batch_size, seq_len] input token sequence
            attention_mask: [batch_size, seq_len] attention mask
            aspect_token_positions: List of (start, end) tuples for known aspect spans
            aspect_mask: [batch_size, seq_len] aspect mask m in {0,1}^n (paper notation)
            labels: [batch_size] labels (optional, for training)
        """
        
        # Filter unsupported arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['num_items_in_batch', 'pos_features']}
        
        # Handle missing aspect parameters for DORA-only configurations
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if aspect_token_positions is None:
            aspect_token_positions = [(0, 1)] * batch_size  # Dummy positions
        if aspect_mask is None:
            aspect_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)  # Empty mask
        
        # ===== Base feature extraction =====
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **filtered_kwargs
        )
        
        # Extract multi-layer hidden states
        all_hidden_states = backbone_outputs.hidden_states  # (num_layers+1, batch, seq, hidden)
        last_hidden = all_hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Use last K layers for deep analysis (K from Phase C sensitivity test)
        layers_to_use = min(self.k_value, len(all_hidden_states) - 1)  # Use configurable K value
        selected_layers = all_hidden_states[-layers_to_use:]  # Last K transformer layers
        
        batch_size, seq_len, hidden_size = last_hidden.shape
        
        # ===== DORA Stage I: shared modeling (Context-Agnostic Multi-Scale Feature Extraction) =====
        
        # DORA 1.1: multi-scale sequence shaping.
        if self.enable_multi_scale:
            enhanced_features = self.dora_multi_scale_sequence_shaping(last_hidden)
        else:
            enhanced_features = last_hidden  # Use original features
        
        # DORA 1.2: inter-layer information flow.
        if self.enable_inter_gru:
            layer_enhanced_features = self.dora_cross_layer_information_flow(selected_layers)
        else:
            layer_enhanced_features = selected_layers  # Use original layers
        
        # ===== ACBS Stage II: aspect-conditioned selection (Aspect-Aware Contextual Focusing) =====

        # Optionally inject mask noise during training (Phase C robustness experiment).
        if self.training and aspect_mask is not None and self.mask_noise_rho > 0:
            # Bernoulli flip some positions in aspect_mask
            noise = (torch.rand_like(aspect_mask.float()) < self.mask_noise_rho).float()
            # Simple flip noise; could later be extended to dilation/erosion.
            aspect_mask = (aspect_mask.float() * (1 - noise) + (1 - aspect_mask.float()) * noise).clamp(0, 1)
            aspect_mask = aspect_mask.long()
        
        # ACBS 2.1: aspect representation.
        need_aspect = (self.enable_aspect_attention or self.enable_context_importance or self.enable_layer_attention)
        aspect_repr = None
        if need_aspect:
            aspect_repr = self.acbs_aspect_representation(
                enhanced_features, aspect_token_positions, aspect_mask
            )
        
        # ACBS 2.2: context reorganization plus aspect-context interaction.
        if self.enable_aspect_attention:
            # Masked self-attention (paper Eq. 154)
            context_enhanced, interaction_weights = self.acbs_aspect_context_mha(
                query=enhanced_features,
                key=enhanced_features, 
                value=enhanced_features,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
        else:
            context_enhanced = enhanced_features  # Use original enhanced features
        
        # ACBS 2.3: sequence-dimension selection.
        if self.enable_context_importance:
            context_importance = self.acbs_sequence_dimension_selection(
                context_enhanced, aspect_repr
            )
        else:
            # Use uniform importance weights
            context_importance = torch.ones_like(context_enhanced[:, :, :1])

        # Exclude padding positions from token selection so c only aggregates over real tokens.
        if attention_mask is not None:
            context_importance = context_importance * attention_mask.unsqueeze(-1).to(context_importance.dtype)

        # ACBS 2.4: layer-dimension selection.
        adaptive_layer_weights = None
        if self.enable_layer_attention:
            # Paper-aligned depth selector: α = Softmax(MLP([a; pool(C)])).
            pooled_context = self._masked_sequence_mean(context_enhanced, attention_mask)
            aspect_context = torch.cat([aspect_repr, pooled_context], dim=-1)  # [batch, hidden*2]
            weights_wide = self.acbs_layer_attention(aspect_context)  # [batch_size, num_layers]
            # Restrict depth selection to the last K substrate layers, then renormalize to a K-way distribution.
            adaptive_layer_weights = weights_wide[:, -layers_to_use:]
            denom = adaptive_layer_weights.sum(dim=-1, keepdim=True).float().clamp_min(1e-6)
            adaptive_layer_weights = (adaptive_layer_weights.float() / denom).to(dtype=weights_wide.dtype)
            weighted_layer_features = self.acbs_layer_dimension_selection(
                layer_enhanced_features, adaptive_layer_weights
            )
        else:
            # Use simple averaging across layers
            if isinstance(layer_enhanced_features, list):
                stacked_features = torch.stack(layer_enhanced_features, dim=2)
                weighted_layer_features = stacked_features.mean(dim=2)
            else:
                weighted_layer_features = layer_enhanced_features
        
        # ACBS 2.5: multi-source fusion (Stage III).
        final_representation, fusion_components = self.acbs_multi_source_fusion(
            context_enhanced, weighted_layer_features, context_importance, aspect_repr, attention_mask=attention_mask
        )
        
        # ===== Classification =====
        logits = self.classifier(final_representation)
        
        # ===== Loss computation (paper Section 3.5) =====
        total_loss = None
        if labels is not None:
            # Class-balanced focal cross-entropy (paper Eqs. 219-221)
            focal_loss = self._compute_focal_loss(logits, labels)
            
            # Regularization terms (paper Section 3.5.2)
            # Regularization terms
            reg_sparse = 0.0
            reg_gate = 0.0
            reg_mask = 0.0

            if not self.disable_regularization:
                reg_sparse = self._compute_sparsity_regularization(context_importance, attention_mask)
                if not self.disable_gate_entropy_reg and not self.disable_gating and 'gate_weights' in fusion_components:
                    reg_gate = self._compute_gate_entropy_regularization(fusion_components['gate_weights'])
                if not self.disable_mask_regularization and aspect_mask is not None and self.enable_context_importance:
                    # R_mask: penalize high activations only inside the aspect span (stable form).
                    # The previous logit(clamp(...)) implementation could overflow to inf in FP16,
                    # then trigger NaNs inside BCEWithLogits. This stable equivalent instead penalizes
                    # the mean gate intensity over the aspect span (L1 on masked gates).
                    m = aspect_mask.float()  # [B, L]
                    denom = m.sum().float().clamp_min(1.0)
                    gates = context_importance.squeeze(-1).float()  # [B, L] in [0,1]
                    reg_mask = (gates * m).sum() / denom

            # Total loss (paper Eqs. 234-238)
            total_loss = (
                focal_loss
                + self.sparse_weight * reg_sparse
                + 1e-2 * reg_gate
                + self.mask_consistency_weight * reg_mask
            )
        
        # Return only essential outputs to avoid None values during evaluation
        result = {
            'loss': total_loss,
            'logits': logits
        }
        
        # Add intermediate results for t-SNE visualization if requested
        if return_intermediates:
            # Extract probability scores
            probs = F.softmax(logits, dim=-1)
            
            # Prepare intermediate results for visualization
            result.update({
                'final_representation': final_representation,  # h: [batch_size, hidden_size]
                'layer_weights': adaptive_layer_weights,  # alpha: [batch_size, K]
                'gate_weights': fusion_components['gate_weights'] if fusion_components and 'gate_weights' in fusion_components else None,  # g: [batch_size, 3]
                'context_importance': context_importance,  # w_list: [batch_size, seq_len, 1]
                'aspect_representation': aspect_repr,  # a: [batch_size, hidden_size]
                'context_representation': fusion_components['c_hat'] if fusion_components and 'c_hat' in fusion_components else None,  # c: [batch_size, hidden_size] 
                'layer_representation': fusion_components['l_hat'] if fusion_components and 'l_hat' in fusion_components else None,  # l: [batch_size, hidden_size]
                'probabilities': probs,  # probs: [batch_size, num_classes]
                'hidden_states': all_hidden_states,  # All layer hidden states for analysis
            })
        
        # Add gate/token statistics for Phase C analysis (only during training to avoid accelerate issues)
        if self.training and fusion_components is not None:
            if 'gate_weights' in fusion_components:
                gate_stats = self.compute_gate_statistics(fusion_components['gate_weights'])
                result['gate_statistics'] = gate_stats
            if context_importance is not None:
                token_stats = self.compute_token_statistics(context_importance, aspect_mask)
                result['token_statistics'] = token_stats
        
        # Only add optional outputs during training to avoid issues with accelerate
        if self.training:
            result.update({
                'hidden_states': all_hidden_states,
                'attentions': backbone_outputs.attentions
            })
        
        return result
    
    def _compute_focal_loss(self, logits, labels, gamma=2.0, alpha=None):
        """
        Class-balanced focal cross-entropy (paper Eqs. 219-221)
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Class weights (alpha)
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                alpha_t = alpha
            else:
                alpha_t = alpha[labels]
        else:
            alpha_t = 1.0
        
        # Focal weight: (1-pt)^gamma
        focal_weight = alpha_t * (1 - pt) ** gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def _compute_sparsity_regularization(self, token_weights, attention_mask):
        """
        R_sparse: L1 regularization applied only to valid tokens (paper Eq. Rsparse)
        """
        # [B, L, 1] * [B, L] -> [B, L]
        w_squeezed = token_weights.squeeze(-1)  # [B, L]
        mask_float = attention_mask.float()  # [B, L]
        # Average only over valid positions.
        valid_sum = (w_squeezed * mask_float).sum()
        valid_count = mask_float.sum()
        return valid_sum / (valid_count + 1e-8)

    def _masked_sequence_mean(self, sequence_features, attention_mask=None):
        """
        Mean pool over valid tokens only when attention_mask is available.
        """
        if attention_mask is None:
            return sequence_features.mean(dim=1)

        mask = attention_mask.to(dtype=sequence_features.dtype).unsqueeze(-1)
        num = (sequence_features * mask).sum(dim=1).float()
        denom = mask.sum(dim=1).float().clamp_min(1e-6)
        return (num / denom).to(dtype=sequence_features.dtype)
    
    def _extract_local_context(self, context_enhanced, context_importance, aspect_token_positions, attention_mask, window=8):
        """
        Extract gate-weighted local context around aspect (±window).
        Different aspect → different gates → different local_ctx → different layer attention

        Args:
            context_enhanced: [batch, seq_len, hidden]
            context_importance: [batch, seq_len, 1] token gates
            aspect_token_positions: List[(start, end)] or None
            attention_mask: [batch, seq_len]
            window: int, context window size (default ±8 tokens)
        Returns:
            local_context: [batch, hidden] gate-weighted local context
        """
        batch_size, seq_len, hidden = context_enhanced.shape
        device = context_enhanced.device
        dtype = context_enhanced.dtype

        if aspect_token_positions is None or len(aspect_token_positions) == 0:
            # Fallback: gate-weighted global mean
            w = context_importance.squeeze(-1)  # [B, L]
            mask = attention_mask.to(dtype=dtype)  # [B, L]
            weights = w * mask  # [B, L]
            # FP16 safety: do normalization in fp32 to avoid eps underflow / div-by-zero.
            denom = weights.sum(dim=1, keepdim=True).float().clamp_min(1e-6)
            weights = (weights.float() / denom).to(dtype=dtype)  # [B, L]
            weighted_ctx = (context_enhanced * weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
            return weighted_ctx

        local_contexts = []
        for i in range(batch_size):
            if i < len(aspect_token_positions) and aspect_token_positions[i] is not None:
                start, end = aspect_token_positions[i]
                # Convert tensor to int if needed
                if isinstance(start, torch.Tensor):
                    start = start.item()
                if isinstance(end, torch.Tensor):
                    end = end.item()
                # Define window boundaries
                win_start = max(0, start - window)
                win_end = min(seq_len, end + window)

                # Create local window mask
                local_mask = torch.zeros(seq_len, device=device, dtype=dtype)
                local_mask[win_start:win_end] = 1.0
                local_mask = local_mask * attention_mask[i].to(dtype=dtype)  # [L]

                # Gate-weighted local context
                w = context_importance[i].squeeze(-1)  # [L]
                p_t = w * local_mask  # [L]
                # FP16 safety: normalize in fp32 to avoid eps underflow / div-by-zero.
                denom = p_t.sum().float().clamp_min(1e-6)
                p_t = (p_t.float() / denom).to(dtype=dtype)  # [L]
                local_ctx = (context_enhanced[i] * p_t.unsqueeze(-1)).sum(dim=0)  # [H]
                local_contexts.append(local_ctx)
            else:
                # Fallback: gate-weighted global mean
                w = context_importance[i].squeeze(-1)  # [L]
                mask = attention_mask[i].to(dtype=dtype)  # [L]
                p_t = w * mask  # [L]
                denom = p_t.sum().float().clamp_min(1e-6)
                p_t = (p_t.float() / denom).to(dtype=dtype)
                local_ctx = (context_enhanced[i] * p_t.unsqueeze(-1)).sum(dim=0)  # [H]
                local_contexts.append(local_ctx)

        return torch.stack(local_contexts, dim=0)  # [batch, hidden]

    def _compute_gate_entropy_regularization(self, gate_weights):
        """
        Gate-entropy regularization (paper Eqs. 202-203, 230)
        """
        # H(g) = -∑ g_i log g_i
        # FP16 safety: clamp in fp32 so 0*log(0) doesn't become NaN.
        gw = gate_weights.float().clamp_min(1e-8)
        entropy = -(gw * torch.log(gw)).sum(dim=-1)
        return -entropy.mean()  # Negative entropy encourages diversity.
    
    def compute_gate_sparsity(self, gate_weights, threshold=0.1):
        """
        Compute sparsity statistics for gate weights.
        
        Args:
            gate_weights: [batch_size, 3] gate weights
            threshold: Sparsity threshold; weights below it are treated as sparse.
        
        Returns:
            sparsity: Sparsity score in [0, 1], where larger means sparser.
        """
        # Method 1: proportion of weights below the threshold.
        sparse_ratio = (gate_weights < threshold).float().mean()
        return sparse_ratio.item()
    
    def compute_gate_statistics(self, gate_weights):
        """
        Compute summary statistics for gate weights.
        
        Returns:
            dict: Statistics including sparsity, mean_weight, std_weight, etc.
        """
        with torch.no_grad():
            stats = {
                'gate_sparsity': self.compute_gate_sparsity(gate_weights),
                'mean_gate_weight': gate_weights.mean().item(),
                'std_gate_weight': gate_weights.std().item(),
                'min_gate_weight': gate_weights.min().item(),
                'max_gate_weight': gate_weights.max().item()
            }
        return stats

    def compute_token_statistics(self, token_weights, aspect_mask=None, threshold=0.5):
        """
        Compute token-gating statistics: mean weight, sparsity, and IoU with the supervision mask when available.
        """
        with torch.no_grad():
            # token_weights: [B, L, 1]
            tw = token_weights.squeeze(-1)
            mean_w = tw.mean().item()
            sparsity = (tw < threshold).float().mean().item()
            stats = {
                'mean_token_weight': mean_w,
                'token_sparsity': sparsity
            }
            if aspect_mask is not None:
                pred_mask = (tw >= threshold).float()
                tgt_mask = aspect_mask.float()
                intersection = (pred_mask * tgt_mask).sum().item()
                union = ((pred_mask + tgt_mask) > 0).float().sum().item()
                iou = (intersection / union) if union > 0 else 0.0
                stats['mask_iou'] = iou
            return stats
    
    def dora_multi_scale_sequence_shaping(self, hidden_states):
        """
        DORA: multi-scale sequence shaping for different semantic granularities (paper Section 3.3.1)
        
        Apply lightweight multi-scale local operators on the top-layer H^{(L)} to capture local compositional patterns.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] top-layer output H^{(L)}
        Returns:
            enhanced_states: [batch_size, seq_len, hidden_size] enhanced representation E
        """
        if not self.enable_multi_scale or self.dora_multi_scale_convs is None:
            return hidden_states
            
        # Transpose for convolution: [batch_size, hidden_size, seq_len]
        x = hidden_states.transpose(1, 2)
        
        # Apply multi-scale convolutions
        multi_scale_features = []
        for conv in self.dora_multi_scale_convs:
            conv_out = F.relu(conv(x))  # [batch_size, hidden_size//3, seq_len]
            multi_scale_features.append(conv_out)
        
        # Concatenate and transpose back
        concatenated = torch.cat(multi_scale_features, dim=1)  # [batch_size, hidden_size, seq_len]
        concatenated = concatenated.transpose(1, 2)  # [batch_size, seq_len, hidden_size]

        # Align with the paper's residual formulation:
        # E = LayerNorm(\tilde{E} W_c + H^(L)).
        projected = self.dora_scale_fusion[0](concatenated)
        enhanced_states = self.dora_scale_fusion[1](projected + hidden_states)
        enhanced_states = self.dora_scale_fusion[3](enhanced_states)

        return enhanced_states
    
    def dora_cross_layer_information_flow(self, selected_layers):
        """
        DORA: inter-layer recursive aggregation via GRU-style information flow (paper Section 3.3.2)

        Implements the paper recurrence:
        s_{1,t} = H^{(L-K+1)}_t
        s_{u,t} = GRUCell(H^{(L-K+u)}_t, s_{u-1,t}), u=2,...,K

        Args:
            selected_layers: List of [batch_size, seq_len, hidden_size] for the last K layers {H^{(L-K+1)}...H^{(L)}}
        Returns:
            processed_layers: List of enhanced layer representations {\tilde{H}^{(k)}}
        """
        if not self.enable_inter_gru:
            return selected_layers

        batch_size, seq_len, hidden_size = selected_layers[0].shape
        num_layers = len(selected_layers)  # K

        # Step 1: Reorder layers for GRU input based on layer_order_mode
        import random
        if self.layer_order_mode == 'reversed':
            layers_for_gru = list(reversed(selected_layers))
            indices = list(reversed(range(num_layers)))
        elif self.layer_order_mode == 'shuffled':
            # Use deterministic random generator to avoid polluting global state
            rng = random.Random(self.shuffle_seed)
            indices = list(range(num_layers))
            rng.shuffle(indices)
            layers_for_gru = [selected_layers[i] for i in indices]
        else:  # normal
            indices = list(range(num_layers))
            layers_for_gru = selected_layers

        # Step 2: Depth integration with selected operator
        mode = (self.inter_layer_fusion or 'gru').strip().lower()
        processed_layers_mode: List[torch.Tensor] = []

        # First step: approximate LayerNorm(beta * s_1 + H_1) with fixed beta=1
        # to keep the implementation close to the paper without adding parameters.
        first_layer = F.layer_norm(
            layers_for_gru[0] + layers_for_gru[0],
            normalized_shape=[hidden_size]
        )
        s_prev = first_layer
        processed_layers_mode.append(first_layer)

        if mode == 'gru':
            if self.dora_inter_gru_cell is None:
                return selected_layers

            for u in range(1, num_layers):
                H_current = layers_for_gru[u]
                # Vectorized GRUCell over (batch*seq) tokens.
                h_in = H_current.reshape(-1, hidden_size)
                h_prev = s_prev.reshape(-1, hidden_size)
                s_flat = self.dora_inter_gru_cell(h_in, h_prev)
                s_current = s_flat.view(batch_size, seq_len, hidden_size)
                s_current_normalized = F.layer_norm(s_current + H_current, normalized_shape=[hidden_size])
                processed_layers_mode.append(s_current_normalized)
                s_prev = s_current_normalized

        elif mode == 'lstm':
            if self.dora_inter_lstm_cell is None:
                return selected_layers

            # Initialize cell state to zeros; hidden state follows s_prev (H_1).
            c_prev = torch.zeros_like(s_prev)
            for u in range(1, num_layers):
                H_current = layers_for_gru[u]
                h_in = H_current.reshape(-1, hidden_size)
                h_prev = s_prev.reshape(-1, hidden_size)
                c_prev_flat = c_prev.reshape(-1, hidden_size)

                h_flat, c_flat = self.dora_inter_lstm_cell(h_in, (h_prev, c_prev_flat))
                h_current = h_flat.view(batch_size, seq_len, hidden_size)
                c_prev = c_flat.view(batch_size, seq_len, hidden_size)

                h_current_norm = F.layer_norm(h_current + H_current, normalized_shape=[hidden_size])
                processed_layers_mode.append(h_current_norm)
                s_prev = h_current_norm

        elif mode in ('cumavg', 'cumulative_avg', 'mean'):
            # Parameter-free ordered integration baseline:
            # mean_prev = mean(H_1..H_{u-1}); s_u = LayerNorm(mean_prev + H_u)
            running_sum = layers_for_gru[0].clone()
            count = 1
            for u in range(1, num_layers):
                H_current = layers_for_gru[u]
                mean_prev = running_sum / float(count)
                s_current = F.layer_norm(mean_prev + H_current, normalized_shape=[hidden_size])
                processed_layers_mode.append(s_current)
                running_sum = running_sum + H_current
                count += 1

        else:
            raise ValueError(f"Unsupported inter_layer_fusion mode: {mode!r}")

        # Step 3: Restore original layer order for layer attention (avoid weight misalignment)
        if self.layer_order_mode == 'normal':
            return processed_layers_mode

        processed_layers = [None] * num_layers
        for i, idx in enumerate(indices):
            processed_layers[idx] = processed_layers_mode[i]
        return processed_layers
    
    def acbs_aspect_representation(self, features, aspect_token_positions=None, aspect_mask=None):
        """
        ACBS: extract the aspect representation (paper Section 3.4.1)
        
        Unified aspect-representation strategy:
        - span-known: use the mean of E_{i:j} as the aspect representation (m_t=1 iff i<=t<=j)
        - span-unknown: use the CLS token or an aspect-text encoding (m=0)
        
        Args:
            features: [batch_size, seq_len, hidden_size] enhanced features E
            aspect_token_positions: List of (start, end) tuples or None
            aspect_mask: [batch_size, seq_len] aspect mask m in {0,1}^n
        Returns:
            aspect_repr: [batch_size, hidden_size] aspect representation a
        """
        batch_size = features.size(0)
        
        # Prefer aspect_mask because it matches the unified paper notation.
        if aspect_mask is not None:
            # Use the mask for weighted averaging (paper Eqs. 146-148).
            aspect_reprs = []
            for i in range(batch_size):
                mask = aspect_mask[i]  # [seq_len]
                if mask.sum() > 0:  # span-known: m != 0
                    # Use the mean of E_{i:j} as the aspect representation.
                    masked_features = features[i] * mask.unsqueeze(-1)  # [seq_len, hidden_size]
                    aspect_repr = masked_features.sum(dim=0) / mask.sum()  # [hidden_size]
                else:  # span-unknown: m = 0
                    # Use alternative pooling if specified, otherwise CLS token
                    if self.pooling_alternative:
                        # Use mean pooling instead of CLS token
                        aspect_repr = features[i].mean(dim=0)  # Mean of all tokens
                    else:
                        # Use the CLS token by default.
                        aspect_repr = features[i, 0, :]
                aspect_reprs.append(aspect_repr)
            return torch.stack(aspect_reprs, dim=0)
        
        # Fallback path: use aspect_token_positions directly.
        if aspect_token_positions is not None:
            aspect_reprs = []
            for i in range(batch_size):
                if (i < len(aspect_token_positions) and 
                    aspect_token_positions[i] is not None):
                    start, end = aspect_token_positions[i]
                    if start >= 0 and end > start and end <= features.size(1):
                        aspect_repr = features[i, start:end, :].mean(dim=0)
                    else:
                        # Use alternative pooling if specified
                        if self.pooling_alternative:
                            aspect_repr = features[i].mean(dim=0)
                        else:
                            aspect_repr = features[i, 0, :]
                else:
                    # Use alternative pooling if specified
                    if self.pooling_alternative:
                        aspect_repr = features[i].mean(dim=0)
                    else:
                        aspect_repr = features[i, 0, :]
                aspect_reprs.append(aspect_repr)
            return torch.stack(aspect_reprs, dim=0)
        
        # Default fallback: use CLS or mean pooling.
        if self.pooling_alternative:
            return features.mean(dim=1)  # [batch_size, hidden_size] Mean pooling
        else:
            return features[:, 0, :]  # [batch_size, hidden_size] CLS token
    
    def acbs_sequence_dimension_selection(self, context_features, aspect_repr):
        """
        ACBS: compute aspect-aware context importance weights (paper Section 3.4.2)
        
        Produce independent gate weights w_t in [0,1] from (C_t, a), yielding an aspect-aware token-selection signal.
        
        Args:
            context_features: [batch_size, seq_len, hidden_size] context representation C
            aspect_repr: [batch_size, hidden_size] aspect representation a
        Returns:
            importance_weights: [batch_size, seq_len, 1] gate weights w_t
        """
        if not self.enable_context_importance or self.acbs_token_gating_net is None:
            # Return uniform weights
            batch_size, seq_len, hidden_size = context_features.shape
            return torch.ones(
                batch_size,
                seq_len,
                1,
                device=context_features.device,
                dtype=context_features.dtype
            )
            
        batch_size, seq_len, hidden_size = context_features.shape
        
        # Expand aspect representation to match sequence length
        aspect_expanded = aspect_repr.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate aspect and context features
        combined_features = torch.cat([context_features, aspect_expanded], dim=-1)
        
        # w_t = sigma(MLP([C_t; a])) in [0,1] (paper Eq. 163)
        importance_weights = self.acbs_token_gating_net(combined_features)
        
        return importance_weights
    
    def acbs_layer_dimension_selection(self, layer_features, layer_weights):
        """
        ACBS: apply adaptive weights to combine multiple layers (paper Section 3.4.3)
        
        Learn an aspect x context-conditioned layer-weight distribution alpha in Delta^{K-1}.
        
        Args:
            layer_features: List of [batch_size, seq_len, hidden_size] enhanced cross-layer features {\tilde{H}^{(k)}}
            layer_weights: [batch_size, K] layer-weight distribution alpha
        Returns:
            weighted_features: [batch_size, seq_len, hidden_size] weighted features
        """
        # Stack layers: [batch_size, seq_len, N, hidden_size]
        stacked_features = torch.stack(layer_features, dim=2)
        
        # Reshape weights for broadcasting: [batch_size, 1, N, 1]
        weights_reshaped = layer_weights.unsqueeze(1).unsqueeze(-1)
        
        # Apply weighted combination
        weighted_features = torch.sum(stacked_features * weights_reshaped, dim=2)
        
        return weighted_features
    
    def acbs_multi_source_fusion(self, context_enhanced, layer_weighted, context_importance, aspect_repr, attention_mask=None):
        """
        ACBS Stage III: fuse all representations into the final feature vector (paper Section 3.2.3)
        
        Implements paper Eqs. 197-198:
        [ĉc, ĉl, ĉa] = LayerNorm([c, l, a])
        [g1, g2, g3] = Softmax(MLP([ĉc;ĉl;ĉa]))
        h = g1*ĉc + g2*ĉl + g3*ĉa
        
        Args:
            context_enhanced: [batch_size, seq_len, hidden_size] context-enhanced features
            layer_weighted: [batch_size, seq_len, hidden_size] layer-weighted features
            context_importance: [batch_size, seq_len, 1] context importance scores
            aspect_repr: [batch_size, hidden_size] aspect representation a
        Returns:
            final_repr: [batch_size, hidden_size] final fused representation h
        """
        # Compute the three source representations.
        # c (token path): pooled context representation.
        # Phase C: context_pooling controls 'mean' vs 'cls'.
        if self.context_pooling == 'cls':
            c = context_enhanced[:, 0, :]  # CLS pooling
        else:
            # Importance-weighted length-invariant mean (paper Eq. 164)
            importance_weighted = context_enhanced * context_importance
            # FP16 safety: compute in fp32 to avoid eps underflow / div-by-zero.
            num = importance_weighted.sum(dim=1).float()
            denom = context_importance.sum(dim=1).float().clamp_min(1e-6)
            c = (num / denom).to(dtype=context_enhanced.dtype)  # [batch_size, hidden_size]
        
        # l (layer path): layer-weighted representation.
        if isinstance(layer_weighted, tuple):
            layer_weighted = layer_weighted[0]
        l = self._masked_sequence_mean(layer_weighted, attention_mask)  # [batch_size, hidden_size]
        
        # Pre-normalization (paper Eqs. 192-194)
        c_hat = F.layer_norm(c, normalized_shape=[self.hidden_size])
        l_hat = F.layer_norm(l, normalized_shape=[self.hidden_size])
        if aspect_repr is None:
            a_hat = torch.zeros_like(c_hat)
        else:
            a_hat = F.layer_norm(aspect_repr, normalized_shape=[self.hidden_size])
        fusion_input = torch.cat([c_hat, l_hat, a_hat], dim=-1)  # [batch_size, 3*hidden_size]
        
        if self.disable_gating:
            # Parameter-free equal fusion; for pure DORA-only (no ACBS), ignore 'a'
            if not (self.enable_aspect_attention or self.enable_context_importance or self.enable_layer_attention):
                final_repr = 0.5 * (c_hat + l_hat)
                gate_weights = torch.tensor(
                    [[0.5, 0.5, 0.0]],
                    device=fusion_input.device,
                    dtype=c_hat.dtype
                ).repeat(c_hat.size(0), 1)
            else:
                final_repr = (c_hat + l_hat + a_hat) / 3.0
                gate_weights = torch.ones(
                    fusion_input.size(0),
                    3,
                    device=fusion_input.device,
                    dtype=c_hat.dtype
                ) / 3.0
        else:
            # Learnable three-source gated fusion (paper Eqs. 197-198)
            gate_weights = self.acbs_fusion_gate(fusion_input)  # [batch_size, 3]
            
            # h = g1*ĉc + g2*ĉl + g3*ĉa
            final_repr = (gate_weights[:, 0:1] * c_hat + 
                         gate_weights[:, 1:2] * l_hat + 
                         gate_weights[:, 2:3] * a_hat)  # [batch_size, hidden_size]
        
        # Return both the fused output and debug components.
        fusion_components = {
            'gate_weights': gate_weights,
            'c_hat': c_hat,
            'l_hat': l_hat, 
            'a_hat': a_hat
        }
        
        return final_repr, fusion_components


def create_dual_layer_model(model_name=config.MODEL_NAME, num_labels=3, **ablation_config):
    """
    Factory function that creates a DORA+ACBS model instance.
    
    Args:
        model_name: Pretrained model name.
        num_labels: Number of target labels.
        **ablation_config: Ablation-study configuration parameters.
    """
    return DualLayerAspectSalienceModel(
        model_name=model_name, 
        num_labels=num_labels,
        **ablation_config
    )
