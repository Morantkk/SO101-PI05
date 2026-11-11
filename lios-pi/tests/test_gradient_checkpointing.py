"""Check recomputation and numerical equivalence of the joint transformer loop on CPU."""

import copy
from types import SimpleNamespace

import torch

from pi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


class TinyJointModel(PaliGemmaWithExpertModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(),
            config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2)),
        )
        self.gemma_expert = SimpleNamespace(model=SimpleNamespace(gradient_checkpointing=False))
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
        self.calls = 0

    def compute_layer_complete(self, layer_idx, inputs_embeds, _attention_mask, _position_ids, _adarms_cond):
        self.calls += 1
        return [torch.sin(self.layers[layer_idx](value)) for value in inputs_embeds]

    def compute_final_norms(self, _models, inputs_embeds, _adarms_cond):
        return inputs_embeds


def test_joint_checkpointing_recomputes_and_preserves_outputs_and_gradients():
    torch.manual_seed(42)
    baseline = TinyJointModel()
    recomputed = copy.deepcopy(baseline)
    recomputed.gemma_expert.model.gradient_checkpointing = True
    inputs = [torch.randn(2, 3, 4), torch.randn(2, 2, 4)]
    baseline_inputs = [value.clone().requires_grad_() for value in inputs]
    checkpoint_inputs = [value.clone().requires_grad_() for value in inputs]
    expected, _ = baseline(inputs_embeds=baseline_inputs)
    actual, _ = recomputed(inputs_embeds=checkpoint_inputs)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right)
    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    assert baseline.calls == 2
    assert recomputed.calls > baseline.calls
    for left, right in zip(baseline.parameters(), recomputed.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)
    for left, right in zip(baseline_inputs, checkpoint_inputs, strict=True):
        torch.testing.assert_close(left.grad, right.grad)


def test_joint_checkpointing_does_not_recompute_during_evaluation():
    model = TinyJointModel().eval()
    model.gemma_expert.model.gradient_checkpointing = True
    outputs, _ = model(inputs_embeds=[torch.randn(2, 3, 4), torch.randn(2, 2, 4)])
    sum(value.square().sum() for value in outputs).backward()
    assert model.calls == 2
