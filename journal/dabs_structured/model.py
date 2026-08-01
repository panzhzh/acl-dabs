#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-encoder DABS model for structured sentiment extraction.

Scope:
  DORA shared substrate -> aspect/opinion BIO proposal heads
  -> Query-Conditioned Budget-Aware Selection over proposed span pairs
  -> NONE/NEG/NEU/POS logits.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .aste.data import ASTE_PAIR_LABEL_TO_ID
from .aste.dataset import IGNORE_INDEX


def _masked_mean(sequence_features, attention_mask=None):
    if attention_mask is None:
        return sequence_features.mean(dim=1)
    mask = attention_mask.to(dtype=sequence_features.dtype).unsqueeze(-1)
    num = (sequence_features * mask).sum(dim=1).float()
    denom = mask.sum(dim=1).float().clamp_min(1e-6)
    return (num / denom).to(dtype=sequence_features.dtype)


class DABSStructuredModel(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_pair_labels: int = len(ASTE_PAIR_LABEL_TO_ID),
        k_value: int = 12,
        dropout: float = 0.1,
        bio_loss_weights: tuple[float, float, float] | None = None,
        pair_loss_weights: tuple[float, float, float, float] | None = None,
        span_proposal_loss_weight: float = 1.0,
        span_proposal_pos_weights: tuple[float, float] | None = None,
        span_proposal_ranking_loss_weight: float = 0.0,
        span_proposal_ranking_margin: float = 1.0,
        span_proposal_ranking_negatives: int = 16,
        pair_head_type: str = "joint",
        pair_relation_loss_weight: float = 1.0,
        pair_polarity_loss_weight: float = 1.0,
        pair_focal_gamma: float = 0.0,
        pair_selection_loss_weight: float = 0.0,
        pair_selection_pos_weight: float | None = None,
        pair_contrastive_loss_weight: float = 0.0,
        pair_contrastive_temperature: float = 0.1,
        pair_distance_embedding_dim: int = 0,
        pair_distance_max: int = 32,
        enable_null_aspects: bool = False,
        num_category_labels: int = 0,
        category_loss_weight: float = 1.0,
        category_pos_weight: Sequence[float] | None = None,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=False,
            torch_dtype=torch.float32,
        )
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.num_layers = int(getattr(self.backbone.config, "num_hidden_layers", 1))
        self.num_pair_labels = int(num_pair_labels)
        # The public journal implementation exposes only the complete DORA--QCBS
        # architecture. Component switches used during research are intentionally
        # absent from this release interface.
        self.proposal_depth_evidence = "span"
        self.pair_depth_evidence = "span"
        self.k_value = int(k_value)
        if self.k_value <= 0:
            raise ValueError("k_value must be positive")
        if self.k_value > self.num_layers:
            raise ValueError(
                f"k_value cannot exceed the {self.num_layers} encoder layers"
            )
        self.span_proposal_loss_weight = float(span_proposal_loss_weight)
        self.span_proposal_ranking_loss_weight = float(span_proposal_ranking_loss_weight)
        self.span_proposal_ranking_margin = float(span_proposal_ranking_margin)
        self.span_proposal_ranking_negatives = int(span_proposal_ranking_negatives)
        self.pair_head_type = pair_head_type.strip().lower()
        if self.pair_head_type not in {"joint", "factorized"}:
            raise ValueError(
                f"Unsupported pair_head_type={pair_head_type!r}; "
                "use 'joint' or 'factorized'."
            )
        self.pair_relation_loss_weight = float(pair_relation_loss_weight)
        self.pair_polarity_loss_weight = float(pair_polarity_loss_weight)
        self.pair_focal_gamma = float(pair_focal_gamma)
        self.pair_selection_loss_weight = float(pair_selection_loss_weight)
        self.pair_contrastive_loss_weight = float(pair_contrastive_loss_weight)
        self.pair_contrastive_temperature = float(pair_contrastive_temperature)
        if self.pair_contrastive_temperature <= 0.0:
            raise ValueError("pair_contrastive_temperature must be positive")
        self.pair_distance_embedding_dim = int(pair_distance_embedding_dim)
        self.pair_distance_max = int(pair_distance_max)
        self.enable_null_aspects = bool(enable_null_aspects)
        self.num_category_labels = int(num_category_labels)
        self.category_loss_weight = float(category_loss_weight)
        if self.num_category_labels < 0:
            raise ValueError("num_category_labels must be non-negative")
        if self.pair_distance_embedding_dim < 0:
            raise ValueError("pair_distance_embedding_dim must be non-negative")
        if self.pair_distance_max <= 0:
            raise ValueError("pair_distance_max must be positive")
        if bio_loss_weights is None:
            self.bio_loss_weights = None
        else:
            self.register_buffer(
                "bio_loss_weights",
                torch.tensor(bio_loss_weights, dtype=torch.float32),
                persistent=False,
            )
        if pair_loss_weights is None:
            self.pair_loss_weights = None
        else:
            self.register_buffer(
                "pair_loss_weights",
                torch.tensor(pair_loss_weights, dtype=torch.float32),
                persistent=False,
            )
        if span_proposal_pos_weights is None:
            self.span_proposal_pos_weights = None
        else:
            self.register_buffer(
                "span_proposal_pos_weights",
                torch.tensor(span_proposal_pos_weights, dtype=torch.float32),
                persistent=False,
            )
        if category_pos_weight is None:
            self.category_pos_weight = None
        else:
            category_pos_weight_tensor = torch.as_tensor(
                tuple(float(value) for value in category_pos_weight),
                dtype=torch.float32,
            )
            if category_pos_weight_tensor.numel() != self.num_category_labels:
                raise ValueError(
                    "category_pos_weight length must equal num_category_labels: "
                    f"{category_pos_weight_tensor.numel()} != "
                    f"{self.num_category_labels}"
                )
            self.register_buffer(
                "category_pos_weight",
                category_pos_weight_tensor,
                persistent=False,
            )
        if pair_selection_pos_weight is None:
            self.pair_selection_pos_weight = None
        else:
            self.register_buffer(
                "pair_selection_pos_weight",
                torch.tensor(float(pair_selection_pos_weight), dtype=torch.float32),
                persistent=False,
            )
        kernels = [1, 3, 5][: max(1, min(3, self.hidden_size))]
        base = self.hidden_size // len(kernels)
        channel_splits = [base] * len(kernels)
        channel_splits[-1] = self.hidden_size - base * (len(kernels) - 1)
        self.dora_multi_scale_convs = nn.ModuleList()
        for kernel_size, out_channels in zip(kernels, channel_splits):
            self.dora_multi_scale_convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=self.hidden_size,
                    ),
                    nn.Conv1d(self.hidden_size, out_channels, kernel_size=1),
                )
            )
        self.dora_scale_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.dora_scale_norm = nn.LayerNorm(self.hidden_size)
        self.dora_scale_dropout = nn.Dropout(dropout)
        self.dora_inter_gru_cell = nn.GRUCell(self.hidden_size, self.hidden_size)

        num_heads = int(getattr(self.backbone.config, "num_attention_heads", 12))
        if self.hidden_size % num_heads != 0:
            num_heads = 1
        self.context_mha = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.aspect_bio_classifier = nn.Linear(self.hidden_size, 3)
        self.opinion_bio_classifier = nn.Linear(self.hidden_size, 3)
        self.proposal_element_embeddings = nn.Parameter(
            torch.empty(2, self.hidden_size)
        )
        nn.init.normal_(self.proposal_element_embeddings, mean=0.0, std=0.02)
        self.proposal_query_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proposal_layer_attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.num_layers),
            nn.Softmax(dim=-1),
        )
        self.proposal_fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.proposal_depth_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proposal_depth_output_weight = nn.Parameter(
            torch.empty(2, self.hidden_size)
        )
        self.proposal_depth_output_bias = nn.Parameter(torch.zeros(2))
        nn.init.normal_(self.proposal_depth_output_weight, mean=0.0, std=0.02)

        self.pair_query_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if self.enable_null_aspects:
            self.null_aspect_embedding = nn.Parameter(torch.empty(self.hidden_size))
            nn.init.normal_(self.null_aspect_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter("null_aspect_embedding", None)
        if self.pair_distance_embedding_dim > 0:
            self.pair_distance_embedding = nn.Embedding(
                self.pair_distance_max * 2 + 1,
                self.pair_distance_embedding_dim,
            )
            self.pair_distance_projection = nn.Linear(
                self.pair_distance_embedding_dim,
                self.hidden_size,
                bias=False,
            )
        else:
            self.pair_distance_embedding = None
            self.pair_distance_projection = None
        self.pair_token_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.pair_layer_attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.num_layers),
            nn.Softmax(dim=-1),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 3),
            nn.Softmax(dim=-1),
        )
        self.pair_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_size // 2,
                self.num_pair_labels
                if self.pair_head_type == "joint"
                else self.num_pair_labels - 1,
            ),
        )
        self.pair_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1),
        )
        self.category_classifier = (
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LayerNorm(self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 2, self.num_category_labels),
            )
            if self.num_category_labels > 0
            else None
        )

    def validate_checkpoint_state_dict(self, state_dict) -> None:
        keys = tuple(state_dict.keys())
        checkpoint_uses_depth_proposals = any(
            key.endswith("proposal_element_embeddings") for key in keys
        )
        checkpoint_uses_last_proposals = any(
            "span_proposal_classifier" in key for key in keys
        )
        checkpoint_uses_pair_distance = any(
            "pair_distance_embedding" in key for key in keys
        )
        if checkpoint_uses_last_proposals:
            raise ValueError(
                "This public model requires a Full DORA--QCBS checkpoint with "
                "depth-selective span proposals."
            )
        if not checkpoint_uses_depth_proposals:
            raise ValueError(
                "Checkpoint does not contain the Full DORA--QCBS span-proposal "
                "readout."
            )
        expected_pair_distance = self.pair_distance_embedding is not None
        if checkpoint_uses_pair_distance != expected_pair_distance:
            raise ValueError(
                "Checkpoint/model pair distance architecture mismatch; set "
                "pair_distance_embedding_dim to the checkpoint value."
            )

    def validate_checkpoint_config(self, config) -> None:
        if not isinstance(config, dict):
            return
        expected = {
            "enable_multi_scale": True,
            "enable_inter_gru": True,
            "enable_context_attention": True,
            "enable_layer_attention": True,
            "enable_pair_token_selection": True,
            "enable_adaptive_fusion": True,
            "depth_order_mode": "normal",
            "single_layer_index": None,
            "k_value": self.k_value,
            "proposal_depth_evidence": self.proposal_depth_evidence,
            "pair_head_type": self.pair_head_type,
            "pair_depth_evidence": self.pair_depth_evidence,
            "pair_distance_embedding_dim": self.pair_distance_embedding_dim,
            "pair_distance_max": self.pair_distance_max,
            "enable_null_aspects": self.enable_null_aspects,
            "num_category_labels": self.num_category_labels,
        }
        mismatches = [
            f"{key}: checkpoint={config[key]!r}, model={value!r}"
            for key, value in expected.items()
            if key in config and config[key] != value
        ]
        if mismatches:
            raise ValueError(
                "Checkpoint/model architecture mismatch: " + "; ".join(mismatches)
            )

    def dora_multi_scale_sequence_shaping(self, last_hidden):
        x = last_hidden.transpose(1, 2)
        features = [F.relu(conv(x)) for conv in self.dora_multi_scale_convs]
        concatenated = torch.cat(features, dim=1).transpose(1, 2)
        projected = self.dora_scale_projection(concatenated)
        enhanced = self.dora_scale_norm(projected + last_hidden)
        return self.dora_scale_dropout(enhanced)

    def dora_cross_layer_information_flow(self, selected_layers):
        batch_size, seq_len, hidden_size = selected_layers[0].shape
        first_layer = F.layer_norm(
            selected_layers[0] + selected_layers[0],
            normalized_shape=[hidden_size],
        )
        processed = [first_layer]
        s_prev = first_layer
        for layer in selected_layers[1:]:
            h_in = layer.reshape(-1, hidden_size)
            h_prev = s_prev.reshape(-1, hidden_size)
            s_flat = self.dora_inter_gru_cell(h_in, h_prev)
            s_current = s_flat.view(batch_size, seq_len, hidden_size)
            s_current = F.layer_norm(s_current + layer, normalized_shape=[hidden_size])
            processed.append(s_current)
            s_prev = s_current
        return processed

    def encode_shared(self, input_ids, attention_mask=None, **kwargs):
        allowed_backbone_keys = {
            "token_type_ids",
            "position_ids",
            "head_mask",
            "inputs_embeds",
            "encoder_hidden_states",
            "encoder_attention_mask",
        }
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k in allowed_backbone_keys},
        )
        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1]
        layers_to_use = min(self.k_value, len(hidden_states) - 1)
        selected_layers = hidden_states[-layers_to_use:]
        selected_layer_indices = tuple(
            range(self.num_layers - layers_to_use + 1, self.num_layers + 1)
        )

        enhanced_features = self.dora_multi_scale_sequence_shaping(last_hidden)
        layer_features = self.dora_cross_layer_information_flow(selected_layers)
        stacked_layers = torch.stack(layer_features, dim=1)
        layer_prefix_features = torch.cat(
            [
                stacked_layers.new_zeros(
                    stacked_layers.shape[0],
                    stacked_layers.shape[1],
                    1,
                    stacked_layers.shape[3],
                ),
                stacked_layers.cumsum(dim=2),
            ],
            dim=2,
        )

        context_features, _ = self.context_mha(
            query=enhanced_features,
            key=enhanced_features,
            value=enhanced_features,
            key_padding_mask=~attention_mask.bool()
            if attention_mask is not None
            else None,
        )

        return {
            "hidden_states": hidden_states,
            "enhanced_features": enhanced_features,
            "context_features": context_features,
            "layer_features": layer_features,
            "layer_prefix_features": layer_prefix_features,
            "layer_pooled_features": [
                _masked_mean(layer, attention_mask) for layer in layer_features
            ],
            "layers_to_use": layers_to_use,
            "selected_layer_indices": selected_layer_indices,
            "depth_order_indices": selected_layer_indices,
            "attention_mask": attention_mask,
        }

    def _span_mean(self, sequence_features, spans, pair_mask):
        # sequence_features: [B, L, H], spans: [B, P, 2]
        batch_size, seq_len, hidden_size = sequence_features.shape
        num_pairs = spans.shape[1]
        if num_pairs == 0:
            return sequence_features.new_zeros((batch_size, 0, hidden_size))

        null_mask = spans[:, :, 0] < 0
        starts = spans[:, :, 0].clamp(min=0, max=max(0, seq_len - 1))
        ends = spans[:, :, 1].clamp(min=1, max=seq_len)
        ends = torch.maximum(ends, starts + 1).clamp(max=seq_len)

        prefix = torch.cat(
            [
                sequence_features.new_zeros((batch_size, 1, hidden_size)),
                sequence_features.cumsum(dim=1),
            ],
            dim=1,
        )
        batch_idx = torch.arange(batch_size, device=sequence_features.device).view(-1, 1)
        batch_idx = batch_idx.expand(-1, num_pairs)
        span_sums = prefix[batch_idx, ends] - prefix[batch_idx, starts]
        span_lengths = (ends - starts).clamp_min(1).to(sequence_features.dtype).unsqueeze(-1)
        span_means = span_sums / span_lengths
        valid_mask = pair_mask.bool() & ~null_mask
        return span_means * valid_mask.to(sequence_features.dtype).unsqueeze(-1)

    def _span_mean_across_layers(
        self,
        layer_features,
        spans,
        pair_mask,
        layer_prefix_features=None,
    ):
        if layer_prefix_features is not None:
            batch_size, num_layers, prefix_len, hidden_size = (
                layer_prefix_features.shape
            )
            seq_len = prefix_len - 1
            num_spans = spans.shape[1]
            if num_spans == 0:
                return layer_prefix_features.new_zeros(
                    (batch_size, 0, num_layers, hidden_size)
                )

            null_mask = spans[:, :, 0] < 0
            starts = spans[:, :, 0].clamp(min=0, max=max(0, seq_len - 1))
            ends = spans[:, :, 1].clamp(min=1, max=seq_len)
            ends = torch.maximum(ends, starts + 1).clamp(max=seq_len)
            batch_idx = torch.arange(
                batch_size, device=layer_prefix_features.device
            ).view(-1, 1, 1)
            layer_idx = torch.arange(
                num_layers, device=layer_prefix_features.device
            ).view(1, -1, 1)
            expanded_starts = starts.unsqueeze(1).expand(-1, num_layers, -1)
            expanded_ends = ends.unsqueeze(1).expand(-1, num_layers, -1)
            span_sums = (
                layer_prefix_features[batch_idx, layer_idx, expanded_ends]
                - layer_prefix_features[batch_idx, layer_idx, expanded_starts]
            )
            span_lengths = (
                (ends - starts)
                .clamp_min(1)
                .to(layer_prefix_features.dtype)
                .unsqueeze(1)
                .unsqueeze(-1)
            )
            span_means = span_sums / span_lengths
            valid_mask = pair_mask.bool() & ~null_mask
            span_means = span_means * valid_mask.to(
                layer_prefix_features.dtype
            ).unsqueeze(1).unsqueeze(-1)
            return span_means.transpose(1, 2)

        stacked = torch.stack(layer_features, dim=1)
        batch_size, num_layers, seq_len, hidden_size = stacked.shape
        num_spans = spans.shape[1]
        flattened = stacked.reshape(batch_size * num_layers, seq_len, hidden_size)
        expanded_spans = (
            spans.unsqueeze(1)
            .expand(-1, num_layers, -1, -1)
            .reshape(batch_size * num_layers, num_spans, 2)
        )
        expanded_mask = (
            pair_mask.unsqueeze(1)
            .expand(-1, num_layers, -1)
            .reshape(batch_size * num_layers, num_spans)
        )
        pooled = self._span_mean(flattened, expanded_spans, expanded_mask)
        return pooled.view(batch_size, num_layers, num_spans, hidden_size).transpose(1, 2)

    def _span_endpoint_representations(self, sequence_features, spans, span_mask):
        batch_size, seq_len, hidden_size = sequence_features.shape
        num_spans = spans.shape[1]
        if num_spans == 0:
            return sequence_features.new_zeros((batch_size, 0, hidden_size * 2))

        null_mask = spans[:, :, 0] < 0
        starts = spans[:, :, 0].clamp(min=0, max=max(0, seq_len - 1))
        ends = spans[:, :, 1].clamp(min=1, max=seq_len)
        ends = torch.maximum(ends, starts + 1).clamp(max=seq_len)
        end_indices = (ends - 1).clamp(min=0, max=max(0, seq_len - 1))
        batch_idx = torch.arange(batch_size, device=sequence_features.device).view(-1, 1)
        batch_idx = batch_idx.expand(-1, num_spans)
        start_repr = sequence_features[batch_idx, starts]
        end_repr = sequence_features[batch_idx, end_indices]
        endpoints = torch.cat([start_repr, end_repr], dim=-1)
        valid_mask = span_mask.bool() & ~null_mask
        return endpoints * valid_mask.to(sequence_features.dtype).unsqueeze(-1)

    def _span_features(self, sequence_features, spans, span_mask):
        span_mean = self._span_mean(sequence_features, spans, span_mask)
        endpoints = self._span_endpoint_representations(
            sequence_features,
            spans,
            span_mask,
        )
        return torch.cat([span_mean, endpoints], dim=-1)

    def span_proposal_readout(
        self,
        shared,
        span_proposal_spans,
        span_proposal_mask,
        return_intermediates=False,
    ):
        enhanced = shared["enhanced_features"]
        span_features = self._span_features(
            enhanced,
            span_proposal_spans,
            span_proposal_mask,
        )
        query = self.proposal_query_mlp(span_features)
        element_query = query.unsqueeze(2) + self.proposal_element_embeddings.view(
            1, 1, 2, self.hidden_size
        )
        pooled_context = _masked_mean(
            shared["context_features"], shared["attention_mask"]
        )
        pooled_context = pooled_context[:, None, None, :].expand(
            -1, element_query.shape[1], 2, -1
        )
        layers_to_use = int(shared["layers_to_use"])
        wide_weights = self.proposal_layer_attention(
            torch.cat([element_query, pooled_context], dim=-1)
        )
        layer_weights = wide_weights[:, :, :, -layers_to_use:]
        layer_weights = (
            layer_weights.float()
            / layer_weights.sum(dim=-1, keepdim=True).float().clamp_min(1e-6)
        ).to(wide_weights.dtype)

        layer_span_features = self._span_mean_across_layers(
            shared["layer_features"],
            span_proposal_spans,
            span_proposal_mask,
            layer_prefix_features=shared.get("layer_prefix_features"),
        )
        depth_summary = torch.sum(
            layer_span_features.unsqueeze(2) * layer_weights.unsqueeze(-1),
            dim=3,
        )
        query_hat = F.layer_norm(element_query, normalized_shape=[self.hidden_size])
        depth_hat = F.layer_norm(depth_summary, normalized_shape=[self.hidden_size])
        depth_gate = self.proposal_fusion_gate(
            torch.cat([query_hat, depth_hat], dim=-1)
        )
        final = depth_gate * depth_hat + (1.0 - depth_gate) * query_hat
        projected = self.proposal_depth_projection(final)
        logits = torch.einsum(
            "bseh,eh->bse", projected, self.proposal_depth_output_weight
        ) + self.proposal_depth_output_bias.view(1, 1, 2)
        logits = logits * span_proposal_mask.to(logits.dtype).unsqueeze(-1)
        if not return_intermediates:
            return logits
        return logits, {
            "proposal_query": element_query,
            "proposal_layer_weights": layer_weights,
            "proposal_depth_gate": depth_gate,
            "proposal_final": final,
        }

    def _span_proposal_ranking_loss(self, span_logits, span_targets, span_mask):
        if self.span_proposal_ranking_loss_weight <= 0.0:
            return span_logits.new_zeros(())

        valid_mask = span_mask.bool()
        losses = []
        for label_idx in range(span_logits.shape[-1]):
            scores = span_logits[:, :, label_idx]
            labels = span_targets[:, :, label_idx] > 0.5
            for batch_idx in range(scores.shape[0]):
                valid = valid_mask[batch_idx]
                pos_scores = scores[batch_idx][valid & labels[batch_idx]]
                neg_scores = scores[batch_idx][valid & ~labels[batch_idx]]
                if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                    continue
                if (
                    self.span_proposal_ranking_negatives > 0
                    and neg_scores.numel() > self.span_proposal_ranking_negatives
                ):
                    neg_scores = torch.topk(
                        neg_scores,
                        k=self.span_proposal_ranking_negatives,
                    ).values
                losses.append(
                    F.relu(
                        self.span_proposal_ranking_margin
                        - pos_scores.unsqueeze(1)
                        + neg_scores.unsqueeze(0)
                    ).mean()
                )

        if not losses:
            return span_logits.new_zeros(())
        return torch.stack(losses).mean()

    def pair_query_readout(
        self,
        shared,
        pair_spans,
        pair_mask,
        return_intermediates=False,
    ):
        context = shared["context_features"]
        enhanced = shared["enhanced_features"]
        attention_mask = shared["attention_mask"]
        layers_to_use = shared["layers_to_use"]

        aspect_spans = pair_spans[:, :, 0:2]
        opinion_spans = pair_spans[:, :, 2:4]
        aspect_null_mask = (aspect_spans[:, :, 0] < 0) & pair_mask.bool()
        opinion_null_mask = (opinion_spans[:, :, 0] < 0) & pair_mask.bool()
        if bool(opinion_null_mask.any()):
            raise ValueError(
                "Negative opinion spans are unsupported: Rest15/Rest16 contain "
                "no implicit opinions."
            )
        if bool(aspect_null_mask.any()) and not self.enable_null_aspects:
            raise ValueError(
                "Negative aspect span indices denote NULL aspects; instantiate "
                "with enable_null_aspects=True."
            )
        aspect_repr = self._span_mean(enhanced, aspect_spans, pair_mask)
        opinion_repr = self._span_mean(enhanced, opinion_spans, pair_mask)
        if self.enable_null_aspects:
            # A bare learned NULL vector is identical for every sentence and
            # leaves implicit-aspect category/polarity prediction largely to
            # the opinion span.  Use the contextualized leading special token
            # as sentence evidence and retain the parameter as a NULL role
            # embedding.  This mirrors how modern ASQP systems represent an
            # implicit target without adding a second encoder pass.
            contextual_null_aspect = F.layer_norm(
                enhanced[:, 0, :] + self.null_aspect_embedding.view(1, -1),
                normalized_shape=[self.hidden_size],
            ).unsqueeze(1)
            aspect_repr = torch.where(
                aspect_null_mask.unsqueeze(-1),
                contextual_null_aspect,
                aspect_repr,
            )
        pair_input = torch.cat(
            [
                aspect_repr,
                opinion_repr,
                torch.abs(aspect_repr - opinion_repr),
                aspect_repr * opinion_repr,
            ],
            dim=-1,
        )
        pair_query = self.pair_query_mlp(pair_input)
        if self.pair_distance_embedding is not None:
            aspect_start, aspect_end, opinion_start, opinion_end = pair_spans.unbind(
                dim=-1
            )
            signed_gap = torch.where(
                opinion_start >= aspect_end,
                opinion_start - aspect_end + 1,
                torch.where(
                    aspect_start >= opinion_end,
                    -(aspect_start - opinion_end + 1),
                    torch.zeros_like(aspect_start),
                ),
            )
            signed_gap = torch.where(
                aspect_null_mask,
                torch.zeros_like(signed_gap),
                signed_gap,
            )
            distance_ids = signed_gap.clamp(
                min=-self.pair_distance_max,
                max=self.pair_distance_max,
            ) + self.pair_distance_max
            distance_features = self.pair_distance_projection(
                self.pair_distance_embedding(distance_ids)
            )
            distance_features = distance_features * pair_mask.to(
                distance_features.dtype
            ).unsqueeze(-1)
            pair_query = F.layer_norm(
                pair_query + distance_features,
                normalized_shape=[self.hidden_size],
            )

        batch_size, seq_len, hidden_size = context.shape
        num_pairs = pair_spans.shape[1]
        context_expanded = context.unsqueeze(1).expand(-1, num_pairs, -1, -1)
        query_expanded = pair_query.unsqueeze(2).expand(-1, -1, seq_len, -1)
        token_gates = self.pair_token_gate(
            torch.cat([context_expanded, query_expanded], dim=-1)
        ).squeeze(-1)
        if attention_mask is not None:
            token_gates = token_gates * attention_mask.unsqueeze(1).to(
                token_gates.dtype
            )
        token_gates = token_gates * pair_mask.unsqueeze(-1).to(token_gates.dtype)
        denom = token_gates.sum(dim=-1, keepdim=True).float().clamp_min(1e-6)
        context_summary = (
            (context_expanded * (token_gates.float() / denom).to(context.dtype).unsqueeze(-1))
            .sum(dim=2)
        )

        pooled_context = _masked_mean(context, attention_mask)
        pooled_context = pooled_context.unsqueeze(1).expand(-1, num_pairs, -1)
        wide_weights = self.pair_layer_attention(
            torch.cat([pair_query, pooled_context], dim=-1)
        )
        layer_weights = wide_weights[:, :, -layers_to_use:]
        layer_weights = (
            layer_weights.float()
            / layer_weights.sum(dim=-1, keepdim=True).float().clamp_min(1e-6)
        ).to(wide_weights.dtype)

        layer_aspect = self._span_mean_across_layers(
            shared["layer_features"],
            aspect_spans,
            pair_mask,
            layer_prefix_features=shared.get("layer_prefix_features"),
        )
        layer_opinion = self._span_mean_across_layers(
            shared["layer_features"],
            opinion_spans,
            pair_mask,
            layer_prefix_features=shared.get("layer_prefix_features"),
        )
        if self.enable_null_aspects and bool(aspect_null_mask.any()):
            contextual_null_layers = torch.stack(
                [layer[:, 0, :] for layer in shared["layer_features"]],
                dim=1,
            )
            contextual_null_layers = F.layer_norm(
                contextual_null_layers
                + self.null_aspect_embedding.view(1, 1, -1),
                normalized_shape=[hidden_size],
            ).unsqueeze(1)
            contextual_null_layers = contextual_null_layers.expand(
                -1, pair_spans.shape[1], -1, -1
            )
            layer_aspect = torch.where(
                aspect_null_mask.unsqueeze(-1).unsqueeze(-1),
                contextual_null_layers,
                layer_aspect,
            )
        stacked_pair_layers = F.layer_norm(
            layer_aspect + layer_opinion + layer_aspect * layer_opinion,
            normalized_shape=[hidden_size],
        )
        layer_summary = torch.sum(
            stacked_pair_layers * layer_weights.unsqueeze(-1),
            dim=2,
        )

        c_hat = F.layer_norm(context_summary, normalized_shape=[hidden_size])
        d_hat = F.layer_norm(layer_summary, normalized_shape=[hidden_size])
        q_hat = F.layer_norm(pair_query, normalized_shape=[hidden_size])
        fusion_weights = self.fusion_gate(
            torch.cat([c_hat, d_hat, q_hat], dim=-1)
        )
        final = (
            fusion_weights[:, :, 0:1] * c_hat
            + fusion_weights[:, :, 1:2] * d_hat
            + fusion_weights[:, :, 2:3] * q_hat
        )
        raw_pair_logits = self.pair_classifier(final)
        pair_selection_logits = self.pair_selector(final).squeeze(-1)
        if self.pair_head_type == "factorized":
            none_log_prob = F.logsigmoid(-pair_selection_logits).unsqueeze(-1)
            relation_log_prob = F.logsigmoid(pair_selection_logits).unsqueeze(-1)
            polarity_log_prob = F.log_softmax(raw_pair_logits, dim=-1)
            pair_logits = torch.cat(
                [none_log_prob, relation_log_prob + polarity_log_prob],
                dim=-1,
            )
        else:
            pair_logits = raw_pair_logits

        if not return_intermediates:
            return pair_logits, {}
        return pair_logits, {
            "pair_query": pair_query,
            "pair_final": final,
            "pair_token_gates": token_gates,
            "pair_layer_weights": layer_weights,
            "pair_fusion_weights": fusion_weights,
            "pair_selection_logits": pair_selection_logits,
            "pair_polarity_logits": raw_pair_logits,
        }

    def _pair_loss(self, pair_logits, pair_labels, pair_loss_weights):
        flat_logits = pair_logits.view(-1, self.num_pair_labels)
        flat_labels = pair_labels.view(-1)
        if self.pair_focal_gamma <= 0.0:
            return F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=IGNORE_INDEX,
                weight=pair_loss_weights,
            )

        valid = flat_labels != IGNORE_INDEX
        if not bool(valid.any()):
            return pair_logits.new_zeros(())
        valid_logits = flat_logits[valid]
        valid_labels = flat_labels[valid]
        ce = F.cross_entropy(
            valid_logits,
            valid_labels,
            weight=pair_loss_weights,
            reduction="none",
        )
        pt = valid_logits.softmax(dim=-1).gather(
            1,
            valid_labels.unsqueeze(1),
        ).squeeze(1)
        focal = (1.0 - pt.clamp(min=1e-6, max=1.0)).pow(self.pair_focal_gamma)
        return (focal * ce).mean()

    def _pair_selection_loss(self, pair_selection_logits, pair_labels, pair_mask):
        valid = pair_mask.bool() & (pair_labels != IGNORE_INDEX)
        if not bool(valid.any()):
            return pair_selection_logits.new_zeros(())
        targets = (pair_labels[valid] != ASTE_PAIR_LABEL_TO_ID["NONE"]).to(
            dtype=pair_selection_logits.dtype
        )
        pos_weight = (
            self.pair_selection_pos_weight.to(
                device=pair_selection_logits.device,
                dtype=pair_selection_logits.dtype,
            )
            if self.pair_selection_pos_weight is not None
            else None
        )
        loss = F.binary_cross_entropy_with_logits(
            pair_selection_logits[valid],
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        if self.pair_focal_gamma > 0.0:
            probabilities = pair_selection_logits[valid].sigmoid()
            pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
            loss = (
                1.0 - pt.clamp(min=1e-6, max=1.0)
            ).pow(self.pair_focal_gamma) * loss
        return loss.mean()

    def _pair_polarity_loss(self, polarity_logits, pair_labels, pair_mask):
        valid = pair_mask.bool() & (pair_labels != IGNORE_INDEX) & (pair_labels != 0)
        if not bool(valid.any()):
            return polarity_logits.new_zeros(())
        labels = pair_labels[valid] - 1
        logits = polarity_logits[valid]
        weights = (
            self.pair_loss_weights[1:].to(
                device=polarity_logits.device,
                dtype=polarity_logits.dtype,
            )
            if self.pair_loss_weights is not None
            else None
        )
        ce = F.cross_entropy(logits, labels, weight=weights, reduction="none")
        if self.pair_focal_gamma > 0.0:
            pt = logits.softmax(dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            ce = (1.0 - pt.clamp(min=1e-6, max=1.0)).pow(
                self.pair_focal_gamma
            ) * ce
        return ce.mean()

    def _pair_contrastive_loss(self, pair_features, pair_labels, pair_mask):
        valid = pair_mask.bool() & (pair_labels != IGNORE_INDEX)
        if int(valid.sum()) < 2:
            return pair_features.new_zeros(())

        features = F.normalize(pair_features[valid].float(), dim=-1)
        labels = pair_labels[valid]
        logits = features @ features.transpose(0, 1)
        logits = logits / self.pair_contrastive_temperature
        self_mask = torch.eye(
            logits.shape[0],
            device=logits.device,
            dtype=torch.bool,
        )
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
        has_positive = positive_mask.any(dim=1)
        if not bool(has_positive.any()):
            return pair_features.new_zeros(())

        log_denominator = torch.logsumexp(
            logits.masked_fill(self_mask, float("-inf")),
            dim=1,
        )
        log_prob = logits - log_denominator.unsqueeze(1)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        anchor_loss = -(
            log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_count
        )
        class_losses = []
        for label in labels.unique():
            class_mask = has_positive & labels.eq(label)
            if bool(class_mask.any()):
                class_losses.append(anchor_loss[class_mask].mean())
        if not class_losses:
            return pair_features.new_zeros(())
        return torch.stack(class_losses).mean()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        aspect_bio_labels=None,
        opinion_bio_labels=None,
        pair_spans=None,
        pair_labels=None,
        pair_mask=None,
        category_targets=None,
        category_target_mask=None,
        span_proposal_spans=None,
        span_aspect_labels=None,
        span_opinion_labels=None,
        span_proposal_mask=None,
        return_intermediates=False,
        **kwargs,
    ):
        shared = self.encode_shared(input_ids, attention_mask=attention_mask, **kwargs)
        context = shared["context_features"]
        aspect_bio_logits = self.aspect_bio_classifier(context)
        opinion_bio_logits = self.opinion_bio_classifier(context)

        if span_proposal_spans is None:
            span_proposal_spans = input_ids.new_zeros((input_ids.shape[0], 0, 2))
        if span_proposal_mask is None:
            span_proposal_mask = torch.zeros(
                span_proposal_spans.shape[:2],
                device=input_ids.device,
                dtype=torch.bool,
            )
        span_proposal_logits = self.span_proposal_readout(
            shared,
            span_proposal_spans=span_proposal_spans,
            span_proposal_mask=span_proposal_mask,
        )

        if pair_spans is None:
            pair_spans = input_ids.new_zeros((input_ids.shape[0], 0, 4))
        if pair_mask is None:
            pair_mask = torch.zeros(pair_spans.shape[:2], device=input_ids.device, dtype=torch.bool)
        need_pair_selection_logits = (
            return_intermediates
            or self.pair_head_type == "factorized"
            or (
                self.pair_contrastive_loss_weight > 0.0
                and pair_labels is not None
            )
            or (
                self.pair_selection_loss_weight > 0.0
                and pair_labels is not None
            )
            or self.category_classifier is not None
        )
        pair_logits, intermediates = self.pair_query_readout(
            shared,
            pair_spans=pair_spans,
            pair_mask=pair_mask,
            return_intermediates=need_pair_selection_logits,
        )
        if self.category_classifier is None:
            category_logits = pair_logits.new_zeros(
                (*pair_logits.shape[:2], 0)
            )
        else:
            pair_features = intermediates.get("pair_final")
            if pair_features is None:
                raise RuntimeError("pair_final is required by the category classifier")
            category_logits = self.category_classifier(pair_features)

        loss = None
        losses = {}
        bio_loss_weights = (
            self.bio_loss_weights.to(
                device=aspect_bio_logits.device,
                dtype=aspect_bio_logits.dtype,
            )
            if self.bio_loss_weights is not None
            else None
        )
        if aspect_bio_labels is not None:
            aspect_loss = F.cross_entropy(
                aspect_bio_logits.view(-1, 3),
                aspect_bio_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                weight=bio_loss_weights,
            )
            losses["aspect_bio_loss"] = aspect_loss
        if opinion_bio_labels is not None:
            opinion_loss = F.cross_entropy(
                opinion_bio_logits.view(-1, 3),
                opinion_bio_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                weight=bio_loss_weights,
            )
            losses["opinion_bio_loss"] = opinion_loss
        if pair_labels is not None and pair_logits.numel() > 0:
            if self.pair_head_type == "factorized":
                selection_logits = intermediates.get("pair_selection_logits")
                polarity_logits = intermediates.get("pair_polarity_logits")
                if selection_logits is None or polarity_logits is None:
                    raise RuntimeError("Factorized pair intermediates are missing.")
                losses["pair_relation_loss"] = (
                    self._pair_selection_loss(selection_logits, pair_labels, pair_mask)
                    * self.pair_relation_loss_weight
                )
                losses["pair_polarity_loss"] = (
                    self._pair_polarity_loss(polarity_logits, pair_labels, pair_mask)
                    * self.pair_polarity_loss_weight
                )
            else:
                pair_loss_weights = (
                    self.pair_loss_weights.to(
                        device=pair_logits.device,
                        dtype=pair_logits.dtype,
                    )
                    if self.pair_loss_weights is not None
                    else None
                )
                pair_loss = self._pair_loss(pair_logits, pair_labels, pair_loss_weights)
                losses["pair_loss"] = pair_loss
                if self.pair_selection_loss_weight > 0.0:
                    selection_logits = intermediates.get("pair_selection_logits")
                    if selection_logits is None:
                        raise RuntimeError(
                            "pair_selection_logits missing for selector loss"
                        )
                    losses["pair_selection_loss"] = (
                        self._pair_selection_loss(
                            selection_logits,
                            pair_labels,
                            pair_mask,
                        )
                        * self.pair_selection_loss_weight
                    )
            if self.pair_contrastive_loss_weight > 0.0:
                pair_features = intermediates.get("pair_final")
                if pair_features is None:
                    raise RuntimeError("pair_final missing for contrastive loss")
                losses["pair_contrastive_loss"] = (
                    self._pair_contrastive_loss(
                        pair_features,
                        pair_labels,
                        pair_mask,
                    )
                    * self.pair_contrastive_loss_weight
                )
        if category_targets is not None:
            if self.category_classifier is None:
                raise ValueError(
                    "category_targets require num_category_labels > 0 when "
                    "constructing the model"
                )
            if tuple(category_targets.shape) != tuple(category_logits.shape):
                raise ValueError(
                    "category_targets shape must match category_logits: "
                    f"{tuple(category_targets.shape)} != "
                    f"{tuple(category_logits.shape)}"
                )
            valid_category_mask = pair_mask.bool()
            if category_target_mask is not None:
                valid_category_mask = (
                    valid_category_mask & category_target_mask.bool()
                )
            if bool(valid_category_mask.any()):
                pos_weight = (
                    self.category_pos_weight.to(
                        device=category_logits.device,
                        dtype=category_logits.dtype,
                    )
                    if self.category_pos_weight is not None
                    else None
                )
                losses["category_loss"] = (
                    F.binary_cross_entropy_with_logits(
                        category_logits[valid_category_mask],
                        category_targets.to(dtype=category_logits.dtype)[
                            valid_category_mask
                        ],
                        pos_weight=pos_weight,
                    )
                    * self.category_loss_weight
                )
        if (
            span_aspect_labels is not None
            and span_opinion_labels is not None
            and span_proposal_logits.numel() > 0
            and bool(span_proposal_mask.any())
        ):
            span_targets = torch.stack(
                [
                    span_aspect_labels.to(dtype=span_proposal_logits.dtype),
                    span_opinion_labels.to(dtype=span_proposal_logits.dtype),
                ],
                dim=-1,
            )
            span_logits_flat = span_proposal_logits[span_proposal_mask]
            span_targets_flat = span_targets[span_proposal_mask]
            pos_weight = (
                self.span_proposal_pos_weights.to(
                    device=span_proposal_logits.device,
                    dtype=span_proposal_logits.dtype,
                )
                if self.span_proposal_pos_weights is not None
                else None
            )
            span_proposal_loss = F.binary_cross_entropy_with_logits(
                span_logits_flat,
                span_targets_flat,
                pos_weight=pos_weight,
            )
            losses["span_proposal_loss"] = (
                span_proposal_loss * self.span_proposal_loss_weight
            )
            span_ranking_loss = self._span_proposal_ranking_loss(
                span_proposal_logits,
                span_targets,
                span_proposal_mask,
            )
            if self.span_proposal_ranking_loss_weight > 0.0:
                losses["span_proposal_ranking_loss"] = (
                    span_ranking_loss * self.span_proposal_ranking_loss_weight
                )

        if losses:
            loss = sum(losses.values())

        result = {
            "loss": loss,
            "aspect_bio_logits": aspect_bio_logits,
            "opinion_bio_logits": opinion_bio_logits,
            "span_proposal_logits": span_proposal_logits,
            "pair_logits": pair_logits,
            "category_logits": category_logits,
        }
        result.update(losses)
        if return_intermediates:
            result.update(intermediates)
        return result
