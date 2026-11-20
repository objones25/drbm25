"""KL-penalised token rewards and GAE, following verl's ``compute_gae_advantage_return``."""

import torch
from torch import Tensor

from drbm25.ppo.losses import validate_token_mask


@torch.no_grad()
def kl_penalized_rewards(
    task_reward: Tensor,
    logprobs_old: Tensor,
    logprobs_ref: Tensor,
    mask: Tensor,
    *,
    kl_coef: float,
) -> Tensor:
    """Per-token reward ``-kl_coef * (old - ref)``, plus ``task_reward`` on each row's last valid token."""
    mask_f = validate_token_mask(mask, logprobs_old=logprobs_old, logprobs_ref=logprobs_ref)
    batch, length = mask_f.shape
    if task_reward.shape != (batch,):
        raise ValueError(f"task_reward has shape {tuple(task_reward.shape)} but expected ({batch},) to match mask rows")
    empty_rows = (mask_f.sum(dim=1) == 0).nonzero()
    if empty_rows.numel() > 0:
        raise ValueError(f"row {int(empty_rows[0].item())} has no valid tokens (every response needs >= 1)")
    rewards = -kl_coef * (logprobs_old - logprobs_ref) * mask_f
    positions = torch.arange(length, device=mask_f.device)
    last_valid = (positions * mask_f.long()).amax(dim=1)
    rewards[torch.arange(batch, device=mask_f.device), last_valid] += task_reward.to(rewards.dtype)
    return rewards


@torch.no_grad()
def gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Return ``(advantages, returns)``, both zero beyond the mask.

    Advantages are *not* whitened here (verl whitens inside its GAE); call
    ``masked_whiten`` on the result. ``returns = advantages + values`` uses the raw advantages.
    """
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError(f"gamma and lam must be in [0, 1], got gamma={gamma}, lam={lam}")
    mask_f = validate_token_mask(mask, rewards=rewards, values=values)
    rewards_m = rewards * mask_f
    values_m = values * mask_f
    length = mask_f.shape[1]
    advantages = torch.zeros_like(values_m)
    last_adv = values_m.new_zeros(values_m.shape[0])
    for t in reversed(range(length)):
        next_value = values_m[:, t + 1] if t + 1 < length else torch.zeros_like(last_adv)
        delta = rewards_m[:, t] + gamma * next_value - values_m[:, t]
        last_adv = delta + gamma * lam * last_adv
        advantages[:, t] = last_adv
    advantages = advantages * mask_f
    returns = (advantages + values_m) * mask_f
    return advantages, returns
