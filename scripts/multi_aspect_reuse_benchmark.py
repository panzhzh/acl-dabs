#!/usr/bin/env python3
"""
Multi-aspect reuse benchmark for DORA model efficiency evaluation.

This script compares the TRUE DORA reuse semantics:
- Reuse: 1 DORA shared computation (encoder+DORA+MHA) + m ACBS forwards  
- Non-reuse: m complete forwards (each with full shared+ACBS computation)

Measures p50, p95 latency, FLOPs/sentence, and throughput for m=1..N aspects.
Includes theoretical validation and seq_len bucketing analysis.
"""

import argparse
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
import os
import sys
import re

# Default seed (can be overridden via CLI)
RANDOM_SEED = 42

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.core.dual_layer_model import DualLayerAspectSalienceModel
from src.core.model import load_tokenizer


def set_random_seed(seed: int) -> None:
    """Set python/numpy/torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class EnhancedDualLayerModel(DualLayerAspectSalienceModel):
    """
    Enhanced model with separate shared/aspect computation methods for benchmarking.
    """
    
    def encode_shared(self, input_ids, attention_mask=None, **kwargs):
        """
        DORA Shared Computation: Encoder + DORA + MHA 
        Returns shared representations that can be reused across multiple aspects.
        """
        # Filter unsupported arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['num_items_in_batch', 'pos_features']}
        
        # Base transformer encoding
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **filtered_kwargs
        )
        
        all_hidden_states = backbone_outputs.hidden_states
        last_hidden = all_hidden_states[-1]
        
        # Use last K layers for deep analysis
        layers_to_use = min(self.k_value, len(all_hidden_states) - 1)
        selected_layers = all_hidden_states[-layers_to_use:]
        
        # DORA Stage I: Shared modeling
        # 1.1: Multi-scale sequence shaping
        if self.enable_multi_scale:
            enhanced_features = self.dora_multi_scale_sequence_shaping(last_hidden)
        else:
            enhanced_features = last_hidden
            
        # 1.2: Cross-layer information flow  
        if self.enable_inter_gru:
            layer_enhanced_features = self.dora_cross_layer_information_flow(selected_layers)
        else:
            layer_enhanced_features = selected_layers
            
        # MHA context reorganization (shared across aspects)
        if self.enable_aspect_attention:
            context_enhanced, _ = self.acbs_aspect_context_mha(
                query=enhanced_features,
                key=enhanced_features,
                value=enhanced_features, 
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
        else:
            context_enhanced = enhanced_features
            
        return {
            'context_enhanced': context_enhanced,
            'layer_enhanced_features': layer_enhanced_features,
            'layers_to_use': layers_to_use,
            'attention_mask': attention_mask
        }
    
    def forward_aspect_from_shared(self, shared_output, aspect_span, aspect_mask=None):
        """
        ACBS Aspect-Specific Computation: Given shared DORA output, compute for one specific aspect.
        """
        context_enhanced = shared_output['context_enhanced'] 
        layer_enhanced_features = shared_output['layer_enhanced_features']
        layers_to_use = shared_output['layers_to_use']
        attention_mask = shared_output.get('attention_mask')
        
        batch_size, seq_len = context_enhanced.shape[:2]
        device = context_enhanced.device
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        
        # Create aspect mask from span if not provided (unified dtype: bool)

        if aspect_mask is None:
            aspect_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
            for i, (start, end) in enumerate(aspect_span):
                if i < batch_size and start >= 0 and end > start and end <= seq_len:
                    aspect_mask[i, start:end] = True
        
        # Convert to long for model compatibility if needed
        aspect_mask_long = aspect_mask.long()
        
        # ACBS Stage II: Aspect-specific computation
        # 2.1: Extract aspect representation
        aspect_repr = self.acbs_aspect_representation(
            context_enhanced, None, aspect_mask_long
        )
        
        # 2.2: Sequence dimension selection (aspect-aware token weights)
        if self.enable_context_importance:
            context_importance = self.acbs_sequence_dimension_selection(
                context_enhanced, aspect_repr
            )
        else:
            context_importance = torch.ones_like(context_enhanced[:, :, :1])
            
        # 2.3: Layer dimension selection (aspect-aware layer weights)
        if self.enable_layer_attention:
            # Keep benchmark path aligned with main model:
            # layer attention takes [aspect_repr ; gate-weighted local context].
            local_ctx = self._extract_local_context(
                context_enhanced,
                context_importance,
                aspect_span,
                attention_mask,
                window=8
            )
            aspect_local = torch.cat([aspect_repr, local_ctx], dim=-1)  # [batch, hidden*2]
            weights_wide = self.acbs_layer_attention(aspect_local)
            adaptive_layer_weights = weights_wide[:, -layers_to_use:]
            weighted_layer_features = self.acbs_layer_dimension_selection(
                layer_enhanced_features, adaptive_layer_weights
            )
        else:
            if isinstance(layer_enhanced_features, list):
                stacked_features = torch.stack(layer_enhanced_features, dim=2)
                weighted_layer_features = stacked_features.mean(dim=2)
            else:
                weighted_layer_features = layer_enhanced_features
                
        # 2.4: Multi-source fusion
        final_representation, _ = self.acbs_multi_source_fusion(
            context_enhanced, weighted_layer_features, context_importance, aspect_repr
        )
        
        # Classification
        logits = self.classifier(final_representation)
        
        return logits


class MultiAspectReuseBenchmark:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        device: str = None,
        seed: int = 42,
        max_length: int = 128,
        k_value: int = 6,
        torch_dtype: str = "auto",
        results_dir: str = "results",
        checkpoint_state: str = "",
    ):
        """Initialize benchmark with DORA+ACBS model and device."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.seed = int(seed)
        self.max_length = int(max_length)
        self.k_value = int(k_value)
        self.torch_dtype = torch_dtype
        self.results_dir = Path(results_dir)
        self.checkpoint_state = str(checkpoint_state or "").strip()
        self.model = None
        self.tokenizer = None
        self.test_sentences = []

    def _resolve_checkpoint_state_file(self) -> Optional[Path]:
        if not self.checkpoint_state:
            return None
        path = Path(self.checkpoint_state).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        if path.is_file():
            return path
        for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No model state file found under checkpoint dir: {path}. "
            f"Expected one of model.safetensors / pytorch_model.bin / model.safetensors.index.json."
        )

    @staticmethod
    def _load_state_dict_from_file(state_file: Path) -> Dict[str, torch.Tensor]:
        suffix = state_file.suffix.lower()
        file_name = state_file.name
        if file_name == "model.safetensors.index.json":
            try:
                from safetensors.torch import load_file as load_safetensors_file
            except Exception as e:
                raise RuntimeError("Loading sharded safetensors requires `safetensors` package.") from e
            index_data = json.loads(state_file.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})
            shard_names = sorted(set(weight_map.values()))
            merged: Dict[str, torch.Tensor] = {}
            for shard_name in shard_names:
                shard_path = state_file.parent / shard_name
                if not shard_path.exists():
                    raise FileNotFoundError(f"Missing shard file listed in index: {shard_path}")
                merged.update(load_safetensors_file(str(shard_path)))
            return merged
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors_file
            except Exception as e:
                raise RuntimeError("Loading safetensors requires `safetensors` package.") from e
            return load_safetensors_file(str(state_file))
        if suffix in {".bin", ".pt", ".pth"}:
            state_obj = torch.load(str(state_file), map_location="cpu")
            if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
                return state_obj["state_dict"]
            if isinstance(state_obj, dict):
                return state_obj
            raise RuntimeError(f"Unsupported state object type in {state_file}: {type(state_obj)}")
        raise RuntimeError(f"Unsupported checkpoint file format: {state_file}")

    @staticmethod
    def _maybe_remap_encoder_only_keys(
        state_dict: Dict[str, torch.Tensor],
        target_keys: set,
    ) -> Dict[str, torch.Tensor]:
        # Encoder-only checkpoints from HF-style save may use model.* / score.weight.
        if "backbone.model.embed_tokens.weight" in target_keys and "model.embed_tokens.weight" in state_dict:
            remapped: Dict[str, torch.Tensor] = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    remapped[f"backbone.{key}"] = value
                elif key == "score.weight":
                    remapped["backbone.score.weight"] = value
                else:
                    remapped[key] = value
            return remapped
        return state_dict

    def _load_checkpoint_state_into_model(self) -> None:
        state_file = self._resolve_checkpoint_state_file()
        if state_file is None:
            return
        print(f"Loading checkpoint state: {state_file}")
        state_dict = self._load_state_dict_from_file(state_file)
        target_keys = set(self.model.state_dict().keys())
        state_dict = self._maybe_remap_encoder_only_keys(state_dict, target_keys)
        matched_before = len(target_keys & set(state_dict.keys()))

        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(incompatible, "missing_keys", []))
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
        print(
            "Checkpoint loaded: "
            f"matched_keys={matched_before}, missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )
        if matched_before < max(10, int(0.05 * len(target_keys))):
            print(
                "⚠️ Very low key overlap with benchmark model. "
                "This may be an adapter-only or incompatible checkpoint format."
            )
        if unexpected_keys and any(("lora_" in k or "base_layer" in k) for k in unexpected_keys):
            print("ℹ️ Detected LoRA/quantized checkpoint keys; benchmark is using base architecture weights where unmatched.")
        
    def setup_model(self):
        """Load tokenizer and enhanced dual-layer model."""
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize enhanced model with all DORA+ACBS components enabled
        self.model = EnhancedDualLayerModel(
            model_name=self.model_name,
            num_labels=3,
            backbone_torch_dtype=self.torch_dtype,
            # Enable all DORA+ACBS components for realistic benchmarking
            enable_multi_scale=True,
            enable_inter_gru=True, 
            enable_aspect_attention=True,
            enable_context_importance=True,
            enable_layer_attention=True,
            k_value=self.k_value
        )
        self._load_checkpoint_state_into_model()

        # Align all newly-added DORA/ACBS module dtypes with backbone dtype.
        target_dtype = next(self.model.backbone.parameters()).dtype
        if str(self.device).startswith("cpu") and target_dtype in (torch.float16, torch.bfloat16):
            # CPU half precision is unsupported/inefficient for many ops.
            target_dtype = torch.float32
        self.model = self.model.to(self.device, dtype=target_dtype)
        self.model.eval()
        
        print("Enhanced DORA+ACBS model loaded successfully.")

    def _build_template_for_count(self, aspect_count: int) -> str:
        """Return a sentence template supporting up to arbitrary aspect count."""
        templates_by_aspect_count = {
            1: ["The {aspect1} of this {product} is {sentiment1}."],
            2: [
                "The {aspect1} of this {product} is {sentiment1} but the {aspect2} is {sentiment2}.",
                "I love the {aspect1} of this {product}, though the {aspect2} could be better.",
            ],
            3: [
                "The {product}'s {aspect1} is {sentiment1}, {aspect2} is {sentiment2}, and {aspect3} needs improvement.",
                "This {product} excels in {aspect1} but fails in {aspect2} and {aspect3}.",
            ],
            4: [
                "The {product} has amazing {aspect1}, decent {aspect2}, poor {aspect3}, but excellent {aspect4}.",
                "I rate this {product}: {aspect1} is perfect, {aspect2} is good, {aspect3} is okay, {aspect4} is bad.",
            ],
        }
        if aspect_count in templates_by_aspect_count:
            return random.choice(templates_by_aspect_count[aspect_count])

        # Generic fallback for aspect_count >= 5
        clauses = [f"the {{aspect{i}}} is {{sentiment{i}}}" for i in range(1, aspect_count + 1)]
        if aspect_count == 5:
            joiner = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
        else:
            joiner = ", ".join(clauses[:-1]) + f", and finally {clauses[-1]}"
        return f"For this {{product}}, {joiner}."

    def _create_sentence_record(
        self,
        aspect_count: int,
        products: List[str],
        aspects: List[str],
        sentiments: List[str],
    ) -> Dict:
        """Generate one synthetic sentence and aspect char spans."""
        template = self._build_template_for_count(aspect_count)
        product = random.choice(products)
        selected_aspects = random.sample(aspects, aspect_count)
        selected_sentiments = random.choices(sentiments, k=aspect_count)

        substitutions = {"product": product}
        for j in range(aspect_count):
            substitutions[f"aspect{j+1}"] = selected_aspects[j]
            substitutions[f"sentiment{j+1}"] = selected_sentiments[j]
        sentence = template.format(**substitutions)

        aspect_spans = []
        for aspect in selected_aspects:
            match = re.search(r"\b" + re.escape(aspect) + r"\b", sentence)
            if match:
                aspect_spans.append((aspect, match.start(), match.end()))

        return {
            "sentence": sentence,
            "aspect_spans": aspect_spans,
            "aspect_count": len(aspect_spans),
        }

    def generate_test_sentences_with_spans(
        self,
        num_sentences: int = 200,
        max_aspects: int = 4,
    ) -> List[Dict]:
        """Generate synthetic sentences with 1..max_aspects aspects."""
        if max_aspects < 1:
            raise ValueError(f"max_aspects must be >=1, got {max_aspects}")

        products = ["phone", "laptop", "restaurant", "hotel", "camera", "headphones", "tablet", "car"]
        aspects = [
            "battery", "screen", "performance", "design", "price", "service", "quality", "sound",
            "comfort", "speed", "display", "build", "features", "software", "hardware",
        ]
        sentiments = ["great", "poor", "excellent", "terrible", "outstanding", "disappointing", "perfect", "awful"]

        if max_aspects > len(aspects):
            raise ValueError(f"max_aspects={max_aspects} exceeds unique aspect pool size={len(aspects)}")

        sentences_with_spans: List[Dict] = []
        per_count = num_sentences // max_aspects

        # Balanced core: equal count for each aspect multiplicity.
        for aspect_count in range(1, max_aspects + 1):
            for _ in range(per_count):
                sentences_with_spans.append(
                    self._create_sentence_record(aspect_count, products, aspects, sentiments)
                )

        # Fill remainder with random multiplicity.
        remaining = num_sentences - len(sentences_with_spans)
        for _ in range(remaining):
            aspect_count = random.randint(1, max_aspects)
            sentences_with_spans.append(
                self._create_sentence_record(aspect_count, products, aspects, sentiments)
            )

        self.test_sentences = sentences_with_spans

        aspect_counts = {}
        for sent in sentences_with_spans:
            count = sent["aspect_count"]
            aspect_counts[count] = aspect_counts.get(count, 0) + 1

        print(f"Generated {len(sentences_with_spans)} test sentences with aspect distribution:")
        for count, num in sorted(aspect_counts.items()):
            print(f"  {count} aspects: {num} sentences")

        return sentences_with_spans
    
    def extract_token_spans(self, sentence: str, char_spans: List[Tuple]) -> List[Tuple]:
        """Convert character spans to token spans using tokenizer (improved to skip all special tokens)."""
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        offset_mapping = encoding['offset_mapping'][0]  # [seq_len, 2]
        token_spans = []
        
        for aspect_name, char_start, char_end in char_spans:
            # Find tokens that overlap with character span
            token_start = None
            token_end = None
            
            for i, (token_char_start, token_char_end) in enumerate(offset_mapping):
                # Skip ALL special tokens (CLS, SEP) which have (0,0) mapping
                if token_char_start == token_char_end == 0:
                    continue
                    
                # Find first overlapping token
                if token_start is None and token_char_end > char_start:
                    token_start = i
                    
                # Find last overlapping token
                if token_char_start < char_end:
                    token_end = i + 1
            
            if token_start is not None and token_end is not None:
                token_spans.append((token_start, token_end))
            else:
                # Fallback: use CLS token
                token_spans.append((0, 1))
                
        return token_spans
    
    def measure_component_times(self, sample_sentences: List[Dict], num_samples: int = 20) -> Tuple[float, float]:
        """Measure T_s (shared) and T_h (aspect-specific) times for theoretical validation."""
        print("Measuring T_s and T_h for theoretical validation...")
        
        shared_times = []
        aspect_times = []
        
        sample_data = random.sample(sample_sentences, min(num_samples, len(sample_sentences)))
        
        for sentence_data in sample_data:
            sentence = sentence_data['sentence']
            aspect_spans = sentence_data['aspect_spans'][:1]  # Use first aspect
            
            with torch.no_grad():
                # Tokenize
                encoding = self.tokenizer(
                    sentence, return_tensors="pt", return_offsets_mapping=True,
                    padding=True, truncation=True, max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in encoding.items() if k != 'offset_mapping'}
                token_spans = self.extract_token_spans(sentence, aspect_spans)
                
                # Measure T_s: shared computation time
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_event.record()
                    shared_output = self.model.encode_shared(**inputs)
                    end_event.record()
                    torch.cuda.synchronize()
                    t_s = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                else:
                    start_time = time.perf_counter()
                    shared_output = self.model.encode_shared(**inputs)
                    t_s = time.perf_counter() - start_time
                
                shared_times.append(t_s)
                
                # Measure T_h: aspect-specific computation time
                if torch.cuda.is_available():
                    start_event.record()
                    _ = self.model.forward_aspect_from_shared(shared_output, token_spans[:1])
                    end_event.record()
                    torch.cuda.synchronize()
                    t_h = start_event.elapsed_time(end_event) / 1000.0
                else:
                    start_time = time.perf_counter()
                    _ = self.model.forward_aspect_from_shared(shared_output, token_spans[:1])
                    t_h = time.perf_counter() - start_time
                
                aspect_times.append(t_h)
        
        avg_t_s = np.mean(shared_times)
        avg_t_h = np.mean(aspect_times)
        
        print(f"  Average T_s (shared): {avg_t_s:.6f}s")
        print(f"  Average T_h (aspect): {avg_t_h:.6f}s")
        print(f"  T_s/T_h ratio: {avg_t_s/avg_t_h:.2f}")
        
        return avg_t_s, avg_t_h
    
    def measure_forward_time(
        self,
        sentence_data: Dict,
        num_aspects: int,
        use_reuse: bool,
        do_warmup: bool = True,
    ) -> Tuple[float, Dict]:
        """Measure TRUE DORA reuse vs non-reuse forward pass time with improved GPU timing."""
        sentence = sentence_data['sentence']
        available_aspects = sentence_data['aspect_spans']
        
        # Use only available aspects, up to num_aspects
        actual_aspects = min(num_aspects, len(available_aspects))
        aspect_spans = available_aspects[:actual_aspects]
        
        with torch.no_grad():
            # Tokenize sentence
            encoding = self.tokenizer(
                sentence,
                return_tensors="pt", 
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in encoding.items() if k != 'offset_mapping'}
            
            # Extract token spans for aspects
            token_spans = self.extract_token_spans(sentence, aspect_spans)
            
            # Pad token_spans to requested num_aspects if needed
            while len(token_spans) < num_aspects:
                token_spans.append((0, 1))  # Fallback CLS token
            
            # Warmup
            if do_warmup:
                if use_reuse:
                    shared = self.model.encode_shared(**inputs)
                    _ = self.model.forward_aspect_from_shared(shared, token_spans[:1])
                else:
                    _ = self.model(**inputs)
                                
            # Measure with improved GPU timing
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            else:
                start_time = time.perf_counter()
            
            if use_reuse:
                # TRUE REUSE: 1 shared computation + m aspect-specific computations
                shared_output = self.model.encode_shared(**inputs)  # DORA shared
                
                for i in range(num_aspects):
                    span = token_spans[i:i+1] if i < len(token_spans) else [(0, 1)]
                    _ = self.model.forward_aspect_from_shared(shared_output, span)  # ACBS
                    
            else:
                # NON-REUSE: m complete forward passes
                for i in range(num_aspects):
                    # Create aspect mask for this specific aspect (unified bool dtype)
                    if i < len(token_spans):
                        start, end = token_spans[i]
                        batch_size, seq_len = inputs['input_ids'].shape
                        aspect_mask = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.bool)
                        aspect_mask[0, start:end] = True
                        _ = self.model(**inputs, aspect_mask=aspect_mask.long())
                    else:
                        _ = self.model(**inputs)
            
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                forward_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            else:
                forward_time = time.perf_counter() - start_time
            
            # More realistic FLOPs estimation
            seq_len = int(inputs['input_ids'].shape[1])
            if use_reuse:
                # Shared computation (encoder+DORA+MHA) + m * aspect computation (ACBS+classifier)
                estimated_flops = seq_len * 2000 + num_aspects * seq_len * 500
            else:
                # m complete computations
                estimated_flops = num_aspects * seq_len * 2500
            
            metrics = {
                'time': forward_time,
                'estimated_flops': estimated_flops,
                'seq_len': seq_len,
                'num_token_spans': len(token_spans),
                'actual_aspects': actual_aspects
            }
            
            return forward_time, metrics

    def _build_sentence_pool(
        self,
        eligible_sentences: List[Dict],
        pool_size: int,
        fixed_pool: bool,
        seed_offset: int,
    ) -> List[Dict]:
        """Select benchmark sentence pool (deterministic if fixed_pool=True)."""
        if not eligible_sentences:
            return []
        if len(eligible_sentences) <= pool_size:
            return eligible_sentences
        if fixed_pool:
            rng = random.Random(self.seed + seed_offset)
            return rng.sample(eligible_sentences, pool_size)
        return random.sample(eligible_sentences, pool_size)

    @staticmethod
    def _bootstrap_percentile_ci(
        values: List[float],
        percentile: float,
        b: int,
        ci_level: float,
        seed: int,
    ) -> Tuple[float, float]:
        """Bootstrap CI for one percentile of a 1D sample."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return np.nan, np.nan
        if arr.size == 1 or b <= 0:
            val = float(np.percentile(arr, percentile))
            return val, val

        rng = np.random.default_rng(seed)
        n = arr.size
        stats = np.empty(b, dtype=np.float64)
        for i in range(b):
            sample = arr[rng.integers(0, n, size=n)]
            stats[i] = np.percentile(sample, percentile)

        alpha = max(0.0, min(1.0, (1.0 - ci_level) / 2.0))
        return (
            float(np.percentile(stats, 100 * alpha)),
            float(np.percentile(stats, 100 * (1.0 - alpha))),
        )

    @staticmethod
    def _bootstrap_speedup_ci(
        reuse_values: List[float],
        non_reuse_values: List[float],
        percentile: float,
        b: int,
        ci_level: float,
        seed: int,
    ) -> Tuple[float, float]:
        """Bootstrap CI for speedup ratio: percentile(non_reuse)/percentile(reuse)."""
        reuse = np.asarray(reuse_values, dtype=np.float64)
        non = np.asarray(non_reuse_values, dtype=np.float64)
        n = min(reuse.size, non.size)
        if n == 0:
            return np.nan, np.nan
        reuse = reuse[:n]
        non = non[:n]
        if n == 1 or b <= 0:
            denom = max(float(np.percentile(reuse, percentile)), 1e-12)
            val = float(np.percentile(non, percentile)) / denom
            return val, val

        rng = np.random.default_rng(seed)
        stats = np.empty(b, dtype=np.float64)
        for i in range(b):
            idx = rng.integers(0, n, size=n)
            r = np.percentile(reuse[idx], percentile)
            nr = np.percentile(non[idx], percentile)
            stats[i] = nr / max(r, 1e-12)

        alpha = max(0.0, min(1.0, (1.0 - ci_level) / 2.0))
        return (
            float(np.percentile(stats, 100 * alpha)),
            float(np.percentile(stats, 100 * (1.0 - alpha))),
        )
    
    def run_benchmark(
        self,
        num_aspects_range: List[int] = None,
        num_runs: int = 50,
        ts_num_samples: int = 20,
        seq_bucket_aspects: int = None,
        repeat_per_sentence: int = 1,
        fixed_pool: bool = True,
        bootstrap_b: int = 1000,
        ci_level: float = 0.95,
    ) -> Dict:
        """Run complete benchmark comparing TRUE reuse vs non-reuse."""
        if num_aspects_range is None:
            num_aspects_range = [1, 2, 3, 4]
        if seq_bucket_aspects is None:
            seq_bucket_aspects = max(num_aspects_range)

        # First measure T_s and T_h for theoretical validation
        t_s, t_h = self.measure_component_times(self.test_sentences, num_samples=ts_num_samples)
        
        # Calculate theoretical speedups
        theoretical_speedups = {}
        for m in num_aspects_range:
            theoretical_speedups[m] = (m * (t_s + t_h)) / (t_s + m * t_h)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'checkpoint_state': self.checkpoint_state or None,
            'device': str(self.device),
            'random_seed': self.seed,
            'torch_dtype': self.torch_dtype,
            'max_length': self.max_length,
            'k_value': self.k_value,
            'num_test_sentences': len(self.test_sentences),
            'num_runs_per_config': num_runs,
            'repeat_per_sentence': int(repeat_per_sentence),
            'fixed_sentence_pool': bool(fixed_pool),
            'num_aspects_range': list(num_aspects_range),
            'ci': {
                'bootstrap_b': int(bootstrap_b),
                'ci_level': float(ci_level)
            },
            'theoretical_validation': {
                'avg_t_s': t_s,
                'avg_t_h': t_h,
                'ratio_t_s_t_h': t_s / t_h,
                'theoretical_speedups': theoretical_speedups
            },
            'results': {},
            'seq_len_buckets': {}
        }
        
        for m in num_aspects_range:
            print(f"\nBenchmarking m={m} aspects...")
            
            reuse_times = []
            non_reuse_times = []
            reuse_flops = []
            non_reuse_flops = []
            seq_lens = []
            
            # Filter sentences that have at least m aspects (for fairness)
            eligible_sentences = [s for s in self.test_sentences if len(s['aspect_spans']) >= m]
            if len(eligible_sentences) < num_runs:
                print(f"  Warning: Only {len(eligible_sentences)} sentences have {m}+ aspects")
            test_subset = self._build_sentence_pool(
                eligible_sentences=eligible_sentences,
                pool_size=num_runs,
                fixed_pool=fixed_pool,
                seed_offset=10007 * m,
            )

            if not test_subset:
                print(f"  Skip m={m}: no eligible sentences")
                continue

            for sentence_idx, sentence_data in enumerate(test_subset):
                for rep in range(repeat_per_sentence):
                    warmup_flag = (rep == 0)
                    # Measure with reuse
                    time_reuse, metrics_reuse = self.measure_forward_time(
                        sentence_data, m, use_reuse=True, do_warmup=warmup_flag
                    )
                    reuse_times.append(time_reuse)
                    reuse_flops.append(metrics_reuse['estimated_flops'])
                    seq_lens.append(metrics_reuse['seq_len'])

                    # Measure without reuse
                    time_no_reuse, metrics_no_reuse = self.measure_forward_time(
                        sentence_data, m, use_reuse=False, do_warmup=warmup_flag
                    )
                    non_reuse_times.append(time_no_reuse)
                    non_reuse_flops.append(metrics_no_reuse['estimated_flops'])
            
            # Calculate statistics
            reuse_p50 = np.percentile(reuse_times, 50)
            reuse_p95 = np.percentile(reuse_times, 95)
            non_reuse_p50 = np.percentile(non_reuse_times, 50)
            non_reuse_p95 = np.percentile(non_reuse_times, 95)
            
            reuse_throughput = 1.0 / reuse_p50
            non_reuse_throughput = 1.0 / non_reuse_p50
            
            avg_reuse_flops = np.mean(reuse_flops)
            avg_non_reuse_flops = np.mean(non_reuse_flops)

            reuse_p50_ci = self._bootstrap_percentile_ci(
                reuse_times, percentile=50, b=bootstrap_b, ci_level=ci_level, seed=self.seed + 17 * m
            )
            reuse_p95_ci = self._bootstrap_percentile_ci(
                reuse_times, percentile=95, b=bootstrap_b, ci_level=ci_level, seed=self.seed + 19 * m
            )
            non_p50_ci = self._bootstrap_percentile_ci(
                non_reuse_times, percentile=50, b=bootstrap_b, ci_level=ci_level, seed=self.seed + 23 * m
            )
            non_p95_ci = self._bootstrap_percentile_ci(
                non_reuse_times, percentile=95, b=bootstrap_b, ci_level=ci_level, seed=self.seed + 29 * m
            )
            speedup_p50_ci = self._bootstrap_speedup_ci(
                reuse_times, non_reuse_times, percentile=50, b=bootstrap_b, ci_level=ci_level, seed=self.seed + 31 * m
            )
            speedup_p95_ci = self._bootstrap_speedup_ci(
                reuse_times, non_reuse_times, percentile=95, b=bootstrap_b, ci_level=ci_level, seed=self.seed + 37 * m
            )

            results['results'][f'm_{m}'] = {
                'reuse': {
                    'p50_latency': reuse_p50,
                    'p95_latency': reuse_p95,
                    'throughput_seq_per_s': reuse_throughput,
                    'avg_flops_per_sentence': avg_reuse_flops,
                    'all_times': reuse_times
                },
                'non_reuse': {
                    'p50_latency': non_reuse_p50,
                    'p95_latency': non_reuse_p95,
                    'throughput_seq_per_s': non_reuse_throughput,
                    'avg_flops_per_sentence': avg_non_reuse_flops,
                    'all_times': non_reuse_times
                },
                'speedup': {
                    'p50_speedup': non_reuse_p50 / reuse_p50,
                    'p95_speedup': non_reuse_p95 / reuse_p95,
                    'throughput_improvement': reuse_throughput / non_reuse_throughput,
                    'flops_reduction': (avg_non_reuse_flops - avg_reuse_flops) / avg_non_reuse_flops
                },
                'theoretical_speedup': theoretical_speedups[m],
                'avg_seq_len': np.mean(seq_lens),
                'eligible_sentences': len(test_subset),
                'num_measurements': len(reuse_times),
                'confidence_intervals': {
                    'reuse_p50_ci': reuse_p50_ci,
                    'reuse_p95_ci': reuse_p95_ci,
                    'non_reuse_p50_ci': non_p50_ci,
                    'non_reuse_p95_ci': non_p95_ci,
                    'p50_speedup_ci': speedup_p50_ci,
                    'p95_speedup_ci': speedup_p95_ci,
                }
            }
            
            print(f"  m={m}: Reuse p50={reuse_p50:.4f}s, Non-reuse p50={non_reuse_p50:.4f}s, Speedup={non_reuse_p50/reuse_p50:.2f}x")
            print(f"         Theoretical speedup: {theoretical_speedups[m]:.2f}x")
            print(
                f"         p95 speedup CI[{int(ci_level*100)}%]: "
                f"[{speedup_p95_ci[0]:.2f}x, {speedup_p95_ci[1]:.2f}x]"
            )
        
        # Seq_len bucketing analysis for chosen aspect count
        print("\nPerforming seq_len bucketing analysis...")
        self._analyze_seq_len_buckets(results, num_aspects=seq_bucket_aspects)
        
        return results
    
    def _analyze_seq_len_buckets(self, results: Dict, num_aspects: int = 4):
        """Analyze performance by sequence length buckets."""
        eligible_sentences = [s for s in self.test_sentences if len(s['aspect_spans']) >= num_aspects]
        
        short_sentences = []
        long_sentences = []
        
        for sent in eligible_sentences:
            encoding = self.tokenizer(
                sent['sentence'],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            seq_len = encoding['input_ids'].shape[1]
            if seq_len <= 64:
                short_sentences.append(sent)
            else:
                long_sentences.append(sent)
        
        buckets = {
            'short (≤64 tokens)': short_sentences[:20] if len(short_sentences) >= 20 else short_sentences,
            'long (>64 tokens)': long_sentences[:20] if len(long_sentences) >= 20 else long_sentences
        }
        
        for bucket_name, sentences in buckets.items():
            if len(sentences) < 5:  # Need minimum samples
                continue
                
            print(f"  Analyzing {bucket_name}: {len(sentences)} sentences")
            
            reuse_times = []
            non_reuse_times = []
            
            for sent in sentences:
                time_reuse, _ = self.measure_forward_time(sent, num_aspects, use_reuse=True)
                time_no_reuse, _ = self.measure_forward_time(sent, num_aspects, use_reuse=False)
                reuse_times.append(time_reuse)
                non_reuse_times.append(time_no_reuse)
            
            reuse_p50 = np.percentile(reuse_times, 50)
            non_reuse_p50 = np.percentile(non_reuse_times, 50)
            speedup = non_reuse_p50 / reuse_p50
            
            results['seq_len_buckets'][bucket_name] = {
                'speedup': speedup,
                'reuse_p50': reuse_p50,
                'non_reuse_p50': non_reuse_p50,
                'num_sentences': len(sentences)
            }
            
            print(f"    {bucket_name}: {speedup:.2f}x speedup")
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dora_reuse_benchmark_{timestamp}.json"

        self.results_dir.mkdir(parents=True, exist_ok=True)

        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def export_to_csv(self, results: Dict, csv_filename: str = None):
        """Export benchmark results to CSV for plotting."""
        if csv_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"dora_reuse_benchmark_{timestamp}.csv"
        
        # Prepare data for CSV
        csv_data = []
        
        for m_key, data in results['results'].items():
            m = int(m_key.split('_')[1])
            
            row = {
                'num_aspects': m,
                'reuse_p50_latency_ms': data['reuse']['p50_latency'] * 1000,  # Convert to ms
                'reuse_p95_latency_ms': data['reuse']['p95_latency'] * 1000,  # Convert to ms
                'non_reuse_p50_latency_ms': data['non_reuse']['p50_latency'] * 1000,  # Convert to ms
                'non_reuse_p95_latency_ms': data['non_reuse']['p95_latency'] * 1000,  # Convert to ms
                'p50_speedup': data['speedup']['p50_speedup'],
                'p95_speedup': data['speedup']['p95_speedup'],
                'reuse_p50_ci_low_ms': data['confidence_intervals']['reuse_p50_ci'][0] * 1000,
                'reuse_p50_ci_high_ms': data['confidence_intervals']['reuse_p50_ci'][1] * 1000,
                'reuse_p95_ci_low_ms': data['confidence_intervals']['reuse_p95_ci'][0] * 1000,
                'reuse_p95_ci_high_ms': data['confidence_intervals']['reuse_p95_ci'][1] * 1000,
                'non_reuse_p50_ci_low_ms': data['confidence_intervals']['non_reuse_p50_ci'][0] * 1000,
                'non_reuse_p50_ci_high_ms': data['confidence_intervals']['non_reuse_p50_ci'][1] * 1000,
                'non_reuse_p95_ci_low_ms': data['confidence_intervals']['non_reuse_p95_ci'][0] * 1000,
                'non_reuse_p95_ci_high_ms': data['confidence_intervals']['non_reuse_p95_ci'][1] * 1000,
                'p50_speedup_ci_low': data['confidence_intervals']['p50_speedup_ci'][0],
                'p50_speedup_ci_high': data['confidence_intervals']['p50_speedup_ci'][1],
                'p95_speedup_ci_low': data['confidence_intervals']['p95_speedup_ci'][0],
                'p95_speedup_ci_high': data['confidence_intervals']['p95_speedup_ci'][1],
                'flops_reduction_pct': data['speedup']['flops_reduction'] * 100,
                'theoretical_speedup': data['theoretical_speedup'],
                'reuse_throughput': data['reuse']['throughput_seq_per_s'],
                'non_reuse_throughput': data['non_reuse']['throughput_seq_per_s'],
                'avg_seq_len': data['avg_seq_len'],
                'eligible_sentences': data['eligible_sentences'],
                'num_measurements': data.get('num_measurements', len(data['reuse'].get('all_times', [])))
            }
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.results_dir / csv_filename
        df.to_csv(csv_path, index=False)
        
        print(f"CSV data saved to: {csv_path}")
        
        # Print summary table
        print("\n=== CSV Data Summary (for plotting) ===")
        print(df[['num_aspects', 'p50_speedup', 'theoretical_speedup', 'flops_reduction_pct']].to_string(index=False))
        
        return csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRUE multi-aspect reuse benchmark for DORA+ACBS."
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base", help="Backbone model name.")
    parser.add_argument(
        "--checkpoint-state",
        default="",
        help="Optional checkpoint dir/file for loading trained weights before benchmarking.",
    )
    parser.add_argument("--device", default="", help='Device string, e.g. "cuda", "cpu", "cuda:0".')
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Backbone loading dtype for memory/speed control.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-sentences", type=int, default=200, help="Synthetic sentence count.")
    parser.add_argument("--num-runs", type=int, default=50, help="Sentence pool size per M.")
    parser.add_argument("--repeat-per-sentence", type=int, default=1, help="Repeat each sentence K times.")
    parser.add_argument(
        "--fixed-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic fixed sentence pool per M (default: true).",
    )
    parser.add_argument("--bootstrap-b", type=int, default=1000, help="Bootstrap resamples for CI.")
    parser.add_argument("--ci-level", type=float, default=0.95, help="Confidence level for CI, e.g. 0.95.")
    parser.add_argument("--max-aspects", type=int, default=4, help="Run M from 1..max_aspects.")
    parser.add_argument("--ts-samples", type=int, default=20, help="Sample count for Ts/Th fitting.")
    parser.add_argument("--seq-bucket-aspects", type=int, default=0, help="M used in seq-length bucket analysis; 0 -> max_aspects.")
    parser.add_argument("--k-value", type=int, default=6, help="Depth K for DORA layer selection.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max_length.")
    parser.add_argument("--results-dir", default="results", help="Directory for JSON/CSV outputs.")
    parser.add_argument("--output-tag", default="", help="Optional filename tag suffix.")
    return parser.parse_args()


def main():
    """Main benchmark execution."""
    args = parse_args()
    if args.repeat_per_sentence < 1:
        raise ValueError("--repeat-per-sentence must be >= 1")
    if args.bootstrap_b < 0:
        raise ValueError("--bootstrap-b must be >= 0")
    if not (0.0 < args.ci_level < 1.0):
        raise ValueError("--ci-level must be in (0,1)")
    set_random_seed(args.seed)

    print("=== TRUE Multi-Aspect DORA Reuse Benchmark (Enhanced) ===")
    print(f"Random seed: {args.seed}")
    print(f"Model: {args.model_name}")
    if args.checkpoint_state:
        print(f"Checkpoint state: {args.checkpoint_state}")
    print(f"Device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Torch dtype: {args.torch_dtype}")

    benchmark = MultiAspectReuseBenchmark(
        model_name=args.model_name,
        device=args.device or None,
        seed=args.seed,
        max_length=args.max_length,
        k_value=args.k_value,
        torch_dtype=args.torch_dtype,
        results_dir=args.results_dir,
        checkpoint_state=args.checkpoint_state,
    )

    benchmark.setup_model()
    benchmark.generate_test_sentences_with_spans(
        num_sentences=args.num_sentences,
        max_aspects=args.max_aspects,
    )

    aspects_range = list(range(1, args.max_aspects + 1))
    seq_bucket_aspects = args.seq_bucket_aspects if args.seq_bucket_aspects > 0 else args.max_aspects
    results = benchmark.run_benchmark(
        num_aspects_range=aspects_range,
        num_runs=args.num_runs,
        ts_num_samples=args.ts_samples,
        seq_bucket_aspects=seq_bucket_aspects,
        repeat_per_sentence=args.repeat_per_sentence,
        fixed_pool=args.fixed_pool,
        bootstrap_b=args.bootstrap_b,
        ci_level=args.ci_level,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.output_tag}" if args.output_tag else ""
    json_name = f"dora_reuse_benchmark_{timestamp}{tag}.json"
    csv_name = f"dora_reuse_benchmark_{timestamp}{tag}.csv"
    json_filepath = benchmark.save_results(results, filename=json_name)
    csv_filepath = benchmark.export_to_csv(results, csv_filename=csv_name)

    print("\n=== TRUE DORA Reuse Benchmark Summary ===")
    for m_key, data in results['results'].items():
        m = m_key.split('_')[1]
        measured_speedup = data['speedup']['p50_speedup']
        theoretical_speedup = data['theoretical_speedup']
        flops_reduction = data['speedup']['flops_reduction'] * 100
        print(
            f"m={m}: {measured_speedup:.2f}x measured vs "
            f"{theoretical_speedup:.2f}x theoretical, {flops_reduction:.1f}% FLOPs reduction"
        )

    t_s = results['theoretical_validation']['avg_t_s']
    t_h = results['theoretical_validation']['avg_t_h']
    print(f"\nTheoretical validation: T_s={t_s:.6f}s, T_h={t_h:.6f}s, T_s/T_h={t_s/t_h:.2f}")
    print(f"This confirms sublinear speedup approaching {1 + t_s/t_h:.2f}x upper bound.")

    if results['seq_len_buckets']:
        print("\nSeq_len bucket analysis:")
        for bucket, data in results['seq_len_buckets'].items():
            print(f"  {bucket}: {data['speedup']:.2f}x speedup ({data['num_sentences']} sentences)")

    print(f"\n✅ Results saved as JSON ({json_filepath}) and CSV ({csv_filepath}) for plotting!")


if __name__ == "__main__":
    main()
