from __future__ import annotations

import tempfile
import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from journal.dabs_structured.asqp.data import parse_rest_quad_line
from journal.dabs_structured.aste.data import parse_aste_line
from journal.dabs_structured.checkpoint import atomic_torch_save, load_checkpoint
from journal.dabs_structured.model import DABSStructuredModel


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=12,
            num_hidden_layers=3,
            num_attention_heads=3,
        )
        self.embedding = nn.Embedding(64, 12)
        self.layers = nn.ModuleList(nn.Linear(12, 12) for _ in range(3))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        states = [hidden]
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
            states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(states))


class JournalMainCodeTest(unittest.TestCase):
    def _build_model(self, **kwargs) -> DABSStructuredModel:
        with patch(
            "journal.dabs_structured.model.AutoModel.from_pretrained",
            return_value=_TinyBackbone(),
        ):
            return DABSStructuredModel(
                model_name="local-tiny-backbone",
                k_value=3,
                pair_distance_embedding_dim=4,
                **kwargs,
            )

    @staticmethod
    def _common_batch():
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 0, 0]], dtype=torch.long
        )
        attention_mask = input_ids.ne(0)
        pair_spans = torch.tensor(
            [[[1, 2, 3, 4], [2, 4, 4, 5]], [[1, 2, 2, 3], [2, 3, 3, 4]]],
            dtype=torch.long,
        )
        pair_mask = torch.ones((2, 2), dtype=torch.bool)
        proposal_spans = torch.tensor(
            [[[1, 2], [2, 4]], [[1, 2], [2, 3]]], dtype=torch.long
        )
        proposal_mask = torch.ones((2, 2), dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "aspect_bio_labels": torch.zeros_like(input_ids),
            "opinion_bio_labels": torch.zeros_like(input_ids),
            "pair_spans": pair_spans,
            "pair_labels": torch.tensor([[3, 0], [1, 0]], dtype=torch.long),
            "pair_mask": pair_mask,
            "span_proposal_spans": proposal_spans,
            "span_aspect_labels": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32
            ),
            "span_opinion_labels": torch.tensor(
                [[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
            ),
            "span_proposal_mask": proposal_mask,
        }

    def test_full_aste_forward_and_backward(self):
        model = self._build_model(
            pair_contrastive_loss_weight=0.1,
            span_proposal_ranking_loss_weight=0.5,
        )
        output = model(**self._common_batch(), return_intermediates=True)
        self.assertEqual(tuple(output["pair_logits"].shape), (2, 2, 4))
        self.assertEqual(tuple(output["span_proposal_logits"].shape), (2, 2, 2))
        self.assertEqual(tuple(output["pair_layer_weights"].shape), (2, 2, 3))
        self.assertTrue(bool(torch.isfinite(output["loss"])))
        output["loss"].backward()
        self.assertIsNotNone(model.dora_inter_gru_cell.weight_hh.grad)

    def test_public_model_exposes_no_component_ablation_switches(self):
        parameters = set(inspect.signature(DABSStructuredModel.__init__).parameters)
        forbidden = {
            "enable_multi_scale",
            "enable_inter_gru",
            "enable_context_attention",
            "enable_layer_attention",
            "enable_pair_token_selection",
            "enable_adaptive_fusion",
            "depth_order_mode",
            "single_layer_index",
            "proposal_depth_evidence",
            "pair_depth_evidence",
        }
        self.assertFalse(parameters & forbidden)

    def test_full_asqp_forward_with_null_aspect(self):
        model = self._build_model(
            pair_head_type="factorized",
            enable_null_aspects=True,
            num_category_labels=13,
            category_pos_weight=[1.0] * 13,
            pair_contrastive_loss_weight=0.1,
        )
        batch = self._common_batch()
        batch["pair_spans"] = batch["pair_spans"].clone()
        batch["pair_spans"][0, 0, :2] = -1
        batch["category_targets"] = torch.zeros((2, 2, 13))
        batch["category_targets"][0, 0, 2] = 1.0
        batch["category_target_mask"] = batch["pair_labels"].ne(0)
        output = model(**batch)
        self.assertEqual(tuple(output["pair_logits"].shape), (2, 2, 4))
        self.assertEqual(tuple(output["category_logits"].shape), (2, 2, 13))
        self.assertTrue(bool(torch.isfinite(output["loss"])))

    def test_data_contracts_and_checkpoint_round_trip(self):
        aste = parse_aste_line(
            "The food is very good####[([1], [4], 'POS')]"
        )
        self.assertEqual(aste.triplets[0].as_tuple(), ((1,), (4,), "POS"))
        asqp = parse_rest_quad_line(
            "The food is good####[['food', 'food quality', 'positive', 'good']]"
        )
        self.assertTrue(asqp.annotations[0].is_representable)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            atomic_torch_save(path, {"value": torch.tensor([1, 2, 3])})
            restored = load_checkpoint(path)
            self.assertTrue(torch.equal(restored["value"], torch.tensor([1, 2, 3])))


if __name__ == "__main__":
    unittest.main()
