"""Inference-time RTC, arXiv:2506.07339, equations 2-5.

Prefix weights and guidance follow Physical-Intelligence/real-time-chunking-kinetix
9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b (MIT; docs/rtc_reference/LICENSE).
LiOS time t runs from noise=1 to data=0; paper time tau is 1-t.
"""

import math

import numpy as np
import torch


def prefix_weights(delay, overlap, horizon, *, device=None, dtype=torch.float32):
    if not 0 <= overlap <= horizon or delay < 0:
        raise ValueError("Invalid RTC overlap or delay")
    start = min(delay, overlap)
    indices = torch.arange(horizon, device=device, dtype=dtype)
    weight = ((start - 1 - indices) / (overlap - start + 1) + 1).clamp(0, 1)
    weight = weight * torch.expm1(weight) / math.expm1(1)
    return torch.where(indices < overlap, weight, 0)


def guidance_weight(tau, beta):
    if not 0 <= tau <= 1 or not math.isfinite(beta) or beta <= 0:
        raise ValueError("Invalid RTC time or guidance cap")
    if tau == 0 or tau == 1:
        return beta
    return min(beta, (tau * tau + (1 - tau) ** 2) / (tau * (1 - tau)))


def encode_previous(actions, state, stats, delta_mask, horizon, action_dim):
    """Rebase physical absolute actions into this observation's normalized frame."""
    actions = np.asarray(actions, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(state) or len(actions) > horizon:
        raise ValueError("Invalid previous RTC chunk")
    if not np.isfinite(actions).all() or not np.isfinite(state).all():
        raise ValueError("Non-finite RTC actions or state")
    result = np.zeros((horizon, action_dim), dtype=np.float32)
    delta = actions - np.where(delta_mask, state, 0)[None, :]
    result[: len(actions), : len(state)] = 2 * (delta - stats.q01) / (stats.q99 - stats.q01 + 1e-6) - 1
    return result


def guided_euler(velocity, noise, previous=None, *, delay=0, overlap=0, num_steps=5, beta=5.0, action_dims=None):
    """Full PiGDM VJP through x - t*v(x,t); detach only between Euler steps."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("RTC requires autograd; inference_mode is not supported")
    if num_steps < 1 or noise.ndim != 3:
        raise ValueError("Invalid RTC sampler shape or steps")
    guidance_weight(0, beta)
    if previous is not None and previous.shape != noise.shape:
        raise ValueError("RTC previous chunk shape differs from noise")
    dims = noise.shape[-1] if action_dims is None else action_dims
    if not 0 < dims <= noise.shape[-1]:
        raise ValueError("Invalid RTC action dimensions")
    weights = prefix_weights(delay, overlap, noise.shape[1], device=noise.device, dtype=noise.dtype)
    weights = weights[None, :, None] * (torch.arange(noise.shape[-1], device=noise.device) < dims)
    x = noise.detach()
    guided = previous is not None and overlap > 0
    for step in range(num_steps):
        tau = step / num_steps
        t = 1 - tau
        time = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        if guided:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                v = velocity(x, time)
                denoised = x - t * v
                error = ((previous.detach() - denoised) * weights).detach()
                correction = torch.autograd.grad(denoised, x, grad_outputs=error)[0]
                v = v.detach() - guidance_weight(tau, beta) * correction
        else:
            with torch.no_grad():
                v = velocity(x, time)
        x = (x.detach() - v.detach() / num_steps).detach()
    return x


def sample_actions_rtc(
    model,
    device,
    observation,
    *,
    previous=None,
    delay=0,
    overlap=0,
    num_steps=5,
    beta=5.0,
    action_dims=12,
    noise=None,
    denoise=None,
):
    """Cache the visual/language prefix once; differentiate only the action expert."""
    from pi.models_pytorch.pi0_pytorch import make_att_2d_masks

    if torch.is_inference_mode_enabled():
        raise RuntimeError("RTC must run outside inference_mode")
    with torch.no_grad():
        images, masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
        prefix, pad, attention = model.embed_prefix(images, masks, tokens, token_masks)
        attention = model._prepare_attention_masks_4d(make_att_2d_masks(pad, attention))
        positions = torch.cumsum(pad, dim=1) - 1
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, cache = model.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        if noise is None:
            noise = model.sample_noise((state.shape[0], model.config.action_horizon, model.config.action_dim), device)
    denoise = model.denoise_step if denoise is None else denoise
    return guided_euler(
        lambda x, t: denoise(state, pad, cache, x, t),
        noise,
        previous,
        delay=delay,
        overlap=overlap,
        num_steps=num_steps,
        beta=beta,
        action_dims=action_dims,
    )
