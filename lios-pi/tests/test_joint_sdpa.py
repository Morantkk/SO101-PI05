"""Compare the joint attention backends, including grouped heads and masked tokens."""

import copy
from types import SimpleNamespace

import torch

from pi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


class IdentityNorm(torch.nn.Module):
    def forward(self, value, cond=None):
        del cond
        return value, None


class Rotary(torch.nn.Module):
    def forward(self, value, positions):
        shape = (*positions.shape, value.shape[-1])
        return torch.ones(shape), torch.zeros(shape)


def make_layer():
    attention = torch.nn.Module()
    attention.head_dim = 4
    attention.scaling = 0.5
    attention.num_key_value_groups = 4
    attention.q_proj = torch.nn.Linear(32, 32)
    attention.k_proj = torch.nn.Linear(32, 8)
    attention.v_proj = torch.nn.Linear(32, 8)
    attention.o_proj = torch.nn.Linear(32, 32)
    layer = torch.nn.Module()
    layer.self_attn = attention
    layer.input_layernorm = IdentityNorm()
    layer.post_attention_layernorm = IdentityNorm()
    layer.mlp = torch.nn.Module()
    layer.mlp.up_proj = torch.nn.Linear(32, 32)
    layer.mlp.forward = layer.mlp.up_proj.forward
    return layer


def test_joint_sdpa_preserves_outputs_and_gradients():
    torch.manual_seed(42)
    eager = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    torch.nn.Module.__init__(eager)
    eager.attention_backend = "eager"
    eager.paligemma = torch.nn.Module()
    eager.paligemma.language_model = torch.nn.Module()
    eager.paligemma.language_model.layers = torch.nn.ModuleList([make_layer()])
    eager.paligemma.model = SimpleNamespace(language_model=SimpleNamespace(rotary_emb=Rotary()))
    eager.gemma_expert = torch.nn.Module()
    eager.gemma_expert.model = torch.nn.Module()
    eager.gemma_expert.model.layers = torch.nn.ModuleList([make_layer()])
    sdpa = copy.deepcopy(eager)
    sdpa.attention_backend = "sdpa"
    inputs = [torch.randn(2, 3, 32), torch.randn(2, 2, 32)]
    left_inputs = [value.clone().requires_grad_() for value in inputs]
    right_inputs = [value.clone().requires_grad_() for value in inputs]
    mask = torch.zeros(2, 1, 5, 5)
    mask[:, :, :3, 3:] = torch.finfo(torch.float32).min
    positions = torch.arange(5).expand(2, -1)
    left = eager.compute_layer_complete(0, left_inputs, mask, positions, [None, None])
    right = sdpa.compute_layer_complete(0, right_inputs, mask, positions, [None, None])
    for expected, actual in zip(left, right, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    sum(value.square().mean() for value in left).backward()
    sum(value.square().mean() for value in right).backward()
    for expected, actual in zip(eager.parameters(), sdpa.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
    for expected, actual in zip(left_inputs, right_inputs, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
