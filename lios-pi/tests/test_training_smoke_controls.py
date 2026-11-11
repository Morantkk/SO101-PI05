"""CPU checks for bounded evaluation and checkpoint optimizer restoration."""

import sys
import copy
from pathlib import Path

import torch
import pytest
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
from utils import run_test_evaluation, fsdp_save_model_checkpoint, resume_from_fsdp_model_checkpoint

from pi.ema_model import EMAModel
from pi.training.config import TrainConfig


def test_evaluation_limit_restores_training_mode():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, _observation, actions):
            self.calls += 1
            return actions.square()

    model = Model().train()
    batches = [{"actions": torch.ones(1, 2), "state": torch.zeros(1, 2)} for _ in range(5)]
    loss = run_test_evaluation(model, batches, torch.device("cpu"), 0, True, global_step=20, max_batches=2)
    assert loss == 1.
    assert model.calls == 2
    assert model.training


def test_checkpoint_restores_optimizer_ema_and_completed_step(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
        model.register_parameter("unused", torch.nn.Parameter(torch.ones(2)))
        model[0].requires_grad_(False)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
        ema = EMAModel(model)
        data = torch.ones(2, 3)
        model(data).square().mean().backward()
        optimizer.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        config = TrainConfig(name="tiny", exp_name="smoke", checkpoint_base_dir=str(tmp_path), overwrite=True)
        fsdp_save_model_checkpoint(model, optimizer, config, True, step=20, ema_model=ema)

        restored = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
        restored.register_parameter("unused", torch.nn.Parameter(torch.ones(2)))
        restored[0].requires_grad_(False)
        restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=0.001)
        restored_ema = EMAModel(restored)
        step = resume_from_fsdp_model_checkpoint(restored, restored_optimizer, config.checkpoint_dir, restored_ema)
        assert step == 20
        assert len(restored_optimizer.state) == len(optimizer.state) == 2
        for key, value in expected_optimizer["state"].items():
            for slot in value:
                torch.testing.assert_close(restored_optimizer.state_dict()["state"][key][slot], value[slot])
        for key in ema.shadow:
            torch.testing.assert_close(restored_ema.shadow[key], ema.shadow[key])
        for net, optim in [(model, optimizer), (restored, restored_optimizer)]:
            net(data).square().mean().backward()
            optim.step()
        for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(original, loaded)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("field", ["max_steps", "max_eval_batches"])
def test_invalid_smoke_limit_is_rejected(field):
    with pytest.raises(ValueError, match="positive"):
        TrainConfig(name="tiny", exp_name="smoke", overwrite=True, **{field: 0})
