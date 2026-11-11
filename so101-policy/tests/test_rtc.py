"""CPU checks against RTC equations and asynchronous tick handoff."""

import sys
import math
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import pytest

from pi.models_pytorch.rtc import guided_euler, prefix_weights, encode_previous, guidance_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deployment/so101"))
from rtc_protocol import RTCPlanner, RTCConsumer, validate_rtc_config


@pytest.mark.parametrize("delay,overlap", [(0, 0), (0, 6), (2, 6), (6, 6), (9, 6), (2, 10)])
def test_paper_exponential_mask(delay, overlap):
    start = min(delay, overlap)
    expected = []
    for i in range(10):
        c = min(1, max(0, (overlap - i) / (overlap - start + 1)))
        expected.append(0 if i >= overlap else c * math.expm1(c) / math.expm1(1))
    torch.testing.assert_close(prefix_weights(delay, overlap, 10), torch.tensor(expected, dtype=torch.float32))


def test_guidance_endpoints_and_cap():
    assert guidance_weight(0, 5) == 5
    assert guidance_weight(1, 5) == 5
    assert guidance_weight(0.5, 5) == 2
    assert guidance_weight(0.01, 5) == 5


def test_full_vjp_sign_and_cotangent_against_nonlinear_jacobian():
    noise = torch.tensor([[[0.3, -0.4], [0.8, 0.1]]], dtype=torch.float64)
    target = torch.tensor([[[0.2, 0.7], [-0.1, 0.5]]], dtype=torch.float64)
    matrix = torch.tensor(
        [[0.1, 0.2, 0.0, 0.1], [0.3, 0.2, 0.1, 0.0], [0.1, 0.0, 0.2, 0.1], [0.0, 0.1, 0.3, 0.2]], dtype=torch.float64
    )

    def velocity(x, t):
        return torch.sin(x.flatten() @ matrix.T).reshape_as(x)

    actual = guided_euler(velocity, noise, target, delay=1, overlap=2, num_steps=5)
    expected = noise.flatten().numpy().copy()
    a = matrix.numpy()
    weights = prefix_weights(1, 2, 2, dtype=torch.float64).numpy().repeat(2)
    for step in range(5):
        tau, t = step / 5, 1 - step / 5
        velocity_value = np.sin(a @ expected)
        jacobian = np.eye(4) - t * np.diag(np.cos(a @ expected)) @ a
        error = (target.numpy().flatten() - (expected - t * velocity_value)) * weights
        expected += (-velocity_value + guidance_weight(tau, 5) * jacobian.T @ error) / 5
    np.testing.assert_allclose(actual.flatten().numpy(), expected, rtol=1e-10, atol=1e-10)
    assert not actual.requires_grad


def test_no_overlap_matches_unguided_and_padding_has_no_target():
    noise = torch.randn(1, 3, 4)
    velocity = lambda x, _t: x * 0.2
    baseline = guided_euler(velocity, noise, num_steps=5)
    torch.testing.assert_close(baseline, noise * 0.96**5)
    torch.testing.assert_close(baseline, guided_euler(velocity, noise, torch.zeros_like(noise), overlap=0))
    guided = guided_euler(velocity, noise, torch.ones_like(noise), overlap=3, action_dims=2)
    torch.testing.assert_close(guided[..., 2:], baseline[..., 2:])
    with torch.inference_mode(), pytest.raises(RuntimeError, match="autograd"):
        guided_euler(velocity, noise)


def test_absolute_to_new_delta_frame_roundtrip():
    rng = np.random.default_rng(42)
    state, absolute = rng.normal(size=12), rng.normal(size=(15, 12))
    mask = np.array([True] * 5 + [False] + [True] * 5 + [False])
    stats = SimpleNamespace(q01=np.full(12, -3), q99=np.full(12, 8))
    encoded = encode_previous(absolute, state, stats, mask, 30, 32)
    decoded = (encoded[:15, :12] + 1) / 2 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
    decoded += np.where(mask, state, 0)
    np.testing.assert_allclose(decoded, absolute, atol=1e-6)
    assert not encoded[:, 12:].any()
    assert not encoded[15:].any()


def config():
    return dict(horizon=30, fps=30, rtc=dict(enabled=True, execution_horizon=15, initial_delay_steps=5, delay_buffer=10))


def result(planner, ticket, value=0, stamp=100):
    actions = np.repeat((np.arange(30) + value)[:, None], 12, axis=1)
    return dict(
        rtc=planner.commit(ticket, actions),
        **{"timestamp/ms": stamp * 1000, "delta/ms": 1000 / 30},
        left_arm=dict(actions=actions[:, :6].tolist()),
        right_arm=dict(actions=actions[:, 6:].tolist()),
        control_mode="joint",
        joint_units="degrees",
        gripper_units="0_to_100",
    )


def setup_queue():
    consumer, planner = RTCConsumer(config()), RTCPlanner(config())
    ticket = planner.plan(dict(rtc=consumer.state()))
    message = result(planner, ticket)
    assert consumer.accept(message, 100.1, 99)
    return consumer, planner, message


def test_async_worker_keeps_old_actions_and_swaps_at_t_minus_s():
    c, p, boot = setup_queue()
    assert p.plan(dict(rtc=c.state())) is None
    for i in range(15):
        assert c.pop()[0] == i
    ticket = p.plan(dict(rtc=c.state()))
    np.testing.assert_array_equal(ticket["previous"][:, 0], np.arange(15, 30))
    for i in range(15, 21):
        assert c.pop()[0] == i
    message = result(p, ticket, value=100)
    assert p.plan(dict(rtc=c.state())) is None
    assert c.accept(message, 100.2, 99)
    assert c.state()["delay_steps"] == 6
    assert c.pop()[0] == 106
    assert not c.accept(message, 100.2, 99)
    assert p.plan(dict(rtc=c.state())) is None
    with pytest.raises(ValueError, match="acknowledged"):
        c.accept(boot, 100.2, 99)


def test_underrun_stale_wrong_session_and_delay_feasibility():
    c, p, message = setup_queue()
    other = RTCConsumer(config())
    assert not other.accept(message, 100.1, 99)
    for _ in range(26):
        c.pop()
    with pytest.raises(ValueError, match="overlap"):
        p.plan(dict(rtc=c.state()))
    for _ in range(4):
        c.pop()
    with pytest.raises(ValueError, match="underrun"):
        c.pop()
    c, p, _ = setup_queue()
    for _ in range(15):
        c.pop()
    next_message = result(p, p.plan(dict(rtc=c.state())))
    with pytest.raises(ValueError, match="Expired"):
        c.accept(next_message, 101, 99)
    p = RTCPlanner(config())
    with pytest.raises(ValueError, match="restarted"):
        p.plan(dict(rtc=c.state()))


def test_delay_buffer_uses_last_ten_and_never_negative():
    c, _, _ = setup_queue()
    c.delays.extend([8, 2, 4])
    assert c.state()["delay_steps"] == 8
    c.delays.extend([3] * 10)
    assert c.state()["delay_steps"] == 3


def test_invalid_rtc_options_are_rejected_before_queue_use():
    for key, value in (("initial_delay_steps", 16), ("delay_buffer", 0), ("num_steps", 0),
                       ("prefix_schedule", "linear"), ("max_guidance_weight", float("nan"))):
        cfg = config()
        cfg["rtc"][key] = value
        with pytest.raises(ValueError):
            validate_rtc_config(cfg)


def test_off_mode_rejects_rtc_actions_and_on_mode_rejects_plain_actions():
    from action_protocol import validate_action
    _, _, message = setup_queue()
    cfg = config()
    cfg["rtc"]["enabled"] = False
    with pytest.raises(ValueError, match="RTC mode"):
        validate_action(message, cfg)
    cfg["rtc"]["enabled"] = True
    del message["rtc"]
    with pytest.raises(ValueError, match="RTC mode"):
        validate_action(message, cfg)
