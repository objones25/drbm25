"""Masked statistics and the clipped PPO policy/value losses over response tokens.

All token tensors are ``[B, T]`` with a 0/1 mask (float or bool). The conventions are
those of verl (``verl/trainer/ppo/core_algos.py``, ``verl/utils/torch_functional.py``)
as used by the upstream DeepRetrieval run: token-mean over every valid token in the
micro-batch, KL estimator ``old - ref``, ``masked_whiten`` with Bessel-corrected variance.
Deliberate differences from verl are listed in the Task 7 report.
"""

import torch
from torch import Tensor


def validate_token_mask(mask: Tensor, **same_shape: Tensor) -> Tensor:
    """Return ``mask`` as a float tensor after checking it and every named tensor are ``[B, T]``.

    Raises ``ValueError`` naming the offending tensors and their shapes: a shape
    mismatch is a programmer error and must fail loudly rather than broadcast.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be [B, T], got shape {tuple(mask.shape)}")
    for name, tensor in same_shape.items():
        if tensor.shape != mask.shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)} but mask has shape {tuple(mask.shape)}")
    mask_float = mask.to(torch.float32) if mask.dtype == torch.bool else mask
    if not bool(((mask_float == 0) | (mask_float == 1)).all()):
        raise ValueError("mask must contain only 0 and 1")
    return mask_float


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    mask_f = validate_token_mask(mask, x=x)
    count = mask_f.sum()
    if count == 0:
        raise ValueError("mask has no valid tokens")
    return (x * mask_f).sum() / count


def masked_whiten(x: Tensor, mask: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Zero-mean, unit-variance over valid tokens (verl: unbiased variance, ``rsqrt(var + eps)``)."""
    mask_f = validate_token_mask(mask, x=x)
    count = mask_f.sum()
    if count == 0:
        raise ValueError("mask has no valid tokens")
    if count < 2:
        raise ValueError(f"masked_whiten needs at least 2 valid tokens, got {int(count.item())}")
    mean = masked_mean(x, mask_f)
    variance = masked_mean((x - mean) ** 2, mask_f) * count / (count - 1)
    return (x - mean) * torch.rsqrt(variance + eps)


def policy_loss(
    logprobs_new: Tensor,
    logprobs_old: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip_ratio: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    mask_f = validate_token_mask(mask, logprobs_new=logprobs_new, logprobs_old=logprobs_old, advantages=advantages)
    log_ratio = logprobs_new - logprobs_old.detach()
    ratio = torch.exp(log_ratio)
    adv = advantages.detach()
    losses_unclipped = -adv * ratio
    losses_clipped = -adv * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    loss = masked_mean(torch.maximum(losses_unclipped, losses_clipped), mask_f)
    stats = {
        "clipfrac": masked_mean((losses_clipped > losses_unclipped).to(ratio.dtype), mask_f).detach(),
        "approx_kl": masked_mean(-log_ratio, mask_f).detach(),
    }
    return loss, stats


def value_loss(
    values_new: Tensor,
    values_old: Tensor,
    returns: Tensor,
    mask: Tensor,
    *,
    clip_range: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    mask_f = validate_token_mask(mask, values_new=values_new, values_old=values_old, returns=returns)
    old = values_old.detach()
    target = returns.detach()
    clipped = torch.minimum(torch.maximum(values_new, old - clip_range), old + clip_range)
    losses_unclipped = (values_new - target) ** 2
    losses_clipped = (clipped - target) ** 2
    loss = 0.5 * masked_mean(torch.maximum(losses_unclipped, losses_clipped), mask_f)
    clipfrac = masked_mean((losses_clipped > losses_unclipped).to(values_new.dtype), mask_f)
    return loss, {"vf_clipfrac": clipfrac.detach()}


def entropy_from_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Mean token entropy of ``softmax(logits)`` over valid tokens; ``logits`` is ``[B, T, V]``.

    fp16/bf16 logits are upcast to fp32 first: half-precision softmax loses ~1e-3 nats.
    """
    if logits.ndim != 3 or logits.shape[:2] != mask.shape:
        raise ValueError(
            f"mask has shape {tuple(mask.shape)} but logits has shape {tuple(logits.shape)} "
            "(expected [B, T, V] with mask [B, T])"
        )
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - (probs * logits).sum(dim=-1)
    return masked_mean(entropy, mask)
