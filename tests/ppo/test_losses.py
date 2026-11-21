import math

import pytest
import torch

from drbm25.ppo.losses import (
    entropy_from_logits,
    masked_mean,
    masked_whiten,
    policy_loss,
    validate_token_mask,
    value_loss,
)

F64 = torch.float64


def _t(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(list(rows), dtype=F64)


def _ones(n: int) -> torch.Tensor:
    return torch.ones(1, n, dtype=F64)


def test_masked_mean_hand_value_ignores_masked_tokens() -> None:
    out = masked_mean(_t([1.0, 3.0, 100.0]), _t([1.0, 1.0, 0.0]))
    assert out.item() == pytest.approx(2.0)


def test_masked_mean_accepts_bool_mask() -> None:
    out = masked_mean(_t([1.0, 3.0, 100.0]), torch.tensor([[True, True, False]]))
    assert out.item() == pytest.approx(2.0)


def test_masked_mean_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="mask has no valid tokens"):
        masked_mean(_ones(3), torch.zeros(1, 3, dtype=F64))


def test_masked_mean_rejects_shape_mismatch_naming_tensors() -> None:
    with pytest.raises(ValueError, match=r"mask.*\(1, 2\)"):
        masked_mean(_ones(3), _ones(2))


def test_whiten_has_zero_mean_unit_std_on_mask() -> None:
    x = _t([1.0, 2.0, 3.0, 50.0], [4.0, 9.0, 7.0, -50.0])
    mask = _t([1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0])
    out = masked_whiten(x, mask)
    valid = out[mask.bool()]
    assert valid.mean().item() == pytest.approx(0.0, abs=1e-12)
    assert valid.std(unbiased=True).item() == pytest.approx(1.0, abs=1e-6)


def test_whiten_hand_values_use_unbiased_variance() -> None:
    # values 1,3: mean 2, unbiased var 2 -> (x-2)/sqrt(2+1e-8)
    out = masked_whiten(_t([1.0, 3.0]), _t([1.0, 1.0]))
    expected = 1.0 / math.sqrt(2.0 + 1e-8)
    torch.testing.assert_close(out, _t([-expected, expected]))


def test_whiten_rejects_single_valid_token() -> None:
    with pytest.raises(ValueError, match="at least 2 valid tokens"):
        masked_whiten(_t([1.0, 2.0]), _t([1.0, 0.0]))


def test_whiten_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="mask has no valid tokens"):
        masked_whiten(_ones(2), torch.zeros(1, 2, dtype=F64))


def test_policy_loss_clips_positive_advantage() -> None:
    logp_old = _t([-1.0])
    loss, stats = policy_loss(logp_old + 0.5, logp_old, _t([1.0]), _ones(1), clip_ratio=0.2)
    assert loss.item() == pytest.approx(-1.2)
    assert stats["clipfrac"].item() == pytest.approx(1.0)


def test_policy_loss_clips_negative_advantage() -> None:
    logp_old = _t([-1.0])
    loss, stats = policy_loss(logp_old + math.log(0.5), logp_old, _t([-1.0]), _ones(1), clip_ratio=0.2)
    assert loss.item() == pytest.approx(0.8)
    assert stats["clipfrac"].item() == pytest.approx(1.0)


def test_policy_loss_unclipped_inside_trust_region() -> None:
    logp_old = _t([-1.0])
    loss, stats = policy_loss(logp_old + math.log(1.1), logp_old, _t([1.0]), _ones(1), clip_ratio=0.2)
    assert loss.item() == pytest.approx(-1.1)
    assert stats["clipfrac"].item() == pytest.approx(0.0)


def test_policy_loss_pessimistic_branch_not_clipped_when_ratio_hurts() -> None:
    # A>0 and ratio < 1-eps: the unclipped term -A*ratio is the larger loss, so no clipping.
    logp_old = _t([-1.0])
    loss, stats = policy_loss(logp_old + math.log(0.5), logp_old, _t([1.0]), _ones(1), clip_ratio=0.2)
    assert loss.item() == pytest.approx(-0.5)
    assert stats["clipfrac"].item() == pytest.approx(0.0)


def test_policy_loss_approx_kl_is_old_minus_new_masked_mean() -> None:
    logp_old = _t([-1.0, -1.0, -1.0])
    logp_new = _t([-0.5, -1.5, 5.0])
    _, stats = policy_loss(logp_new, logp_old, _t([1.0, 1.0, 1.0]), _t([1.0, 1.0, 0.0]), clip_ratio=0.2)
    assert stats["approx_kl"].item() == pytest.approx(0.0)
    _, stats = policy_loss(logp_new + 0.1, logp_old, _t([1.0, 1.0, 1.0]), _t([1.0, 1.0, 0.0]), clip_ratio=0.2)
    assert stats["approx_kl"].item() == pytest.approx(-0.1)


def test_policy_loss_is_token_mean_over_all_valid_tokens() -> None:
    # Row 0 has 1 valid token (loss -1), row 1 has 3 (loss -2 each): mean over 4 tokens = -1.75.
    zeros = torch.zeros(2, 3, dtype=F64)
    adv = _t([1.0, 9.0, 9.0], [2.0, 2.0, 2.0])
    mask = _t([1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    loss, _ = policy_loss(zeros, zeros, adv, mask, clip_ratio=0.2)
    assert loss.item() == pytest.approx(-1.75)


def test_policy_loss_gradient_only_flows_to_new_logprobs() -> None:
    logp_new = _t([-1.0, -1.0]).requires_grad_()
    logp_old = _t([-1.0, -1.0]).requires_grad_()
    loss, stats = policy_loss(logp_new, logp_old, _t([1.0, 1.0]), _t([1.0, 0.0]), clip_ratio=0.2)
    loss.backward()
    assert logp_new.grad is not None
    torch.testing.assert_close(logp_new.grad, _t([-1.0, 0.0]))
    assert not stats["clipfrac"].requires_grad
    assert not stats["approx_kl"].requires_grad


def test_value_loss_hand_value() -> None:
    loss, stats = value_loss(_t([1.0]), _t([0.0]), _t([0.2]), _ones(1), clip_range=0.5)
    assert loss.item() == pytest.approx(0.32)
    assert stats["vf_clipfrac"].item() == pytest.approx(0.0)


def test_value_loss_clipped_branch_wins_when_it_is_larger() -> None:
    # v_old=0, v_new=0.4, R=2... use v_new inside range: clipped == new -> no clipfrac.
    # v_old=1, v_new=0, R=1, clip 0.1: clipped=0.9; (0-1)^2=1 vs (0.9-1)^2=0.01 -> unclipped wins.
    loss, stats = value_loss(_t([0.0]), _t([1.0]), _t([1.0]), _ones(1), clip_range=0.1)
    assert loss.item() == pytest.approx(0.5)
    assert stats["vf_clipfrac"].item() == pytest.approx(0.0)
    # v_old=0, v_new=1.0, R=0.2... covered above. Clipped larger: v_old=0, v_new=0.1, R=0.4? no;
    # v_new=-1, v_old=0, R=-0.5, clip 0.1: l1=0.25, clipped=-0.1 -> l2=0.16, unclipped wins again.
    # Clipped larger needs the new value to move toward R beyond the range: v_old=0, v_new=1, R=1:
    # l1=0, clipped=0.1 -> l2=0.81 > l1.
    loss, stats = value_loss(_t([1.0]), _t([0.0]), _t([1.0]), _ones(1), clip_range=0.1)
    assert loss.item() == pytest.approx(0.5 * 0.81)
    assert stats["vf_clipfrac"].item() == pytest.approx(1.0)


def test_value_loss_token_mean_ignores_masked() -> None:
    loss, _ = value_loss(_t([1.0, 50.0]), _t([1.0, 0.0]), _t([0.0, 0.0]), _t([1.0, 0.0]), clip_range=0.5)
    assert loss.item() == pytest.approx(0.5)


def test_value_loss_gradient_flows_to_values_new_only() -> None:
    v_new = _t([1.0]).requires_grad_()
    v_old = _t([0.0]).requires_grad_()
    loss, _ = value_loss(v_new, v_old, _t([0.2]), _ones(1), clip_range=0.5)
    loss.backward()
    assert v_new.grad is not None
    torch.testing.assert_close(v_new.grad, _t([0.8]))


def test_entropy_uniform_logits_equals_log_vocab() -> None:
    out = entropy_from_logits(torch.zeros(2, 3, 8, dtype=F64), torch.ones(2, 3, dtype=F64))
    assert out.item() == pytest.approx(math.log(8))


def test_entropy_one_hot_like_logits_is_near_zero_and_masked() -> None:
    logits = torch.zeros(1, 2, 4, dtype=F64)
    logits[0, 0, 0] = 100.0  # token 0 nearly deterministic, token 1 uniform (masked out)
    out = entropy_from_logits(logits, _t([1.0, 0.0]))
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_entropy_is_stable_for_large_logits() -> None:
    logits = torch.tensor([[[1e4, 1e4, 0.0]]], dtype=F64)
    out = entropy_from_logits(logits, _ones(1))
    assert out.item() == pytest.approx(math.log(2))


def test_entropy_rejects_mask_not_matching_logits_prefix() -> None:
    with pytest.raises(ValueError, match=r"mask.*\(1, 2\).*logits.*\(1, 3, 4\)"):
        entropy_from_logits(torch.zeros(1, 3, 4), _ones(2))


def test_whiten_eps_enters_the_denominator() -> None:
    out = masked_whiten(_t([1.0, 3.0]), _t([1.0, 1.0]), eps=1.0)  # var 2 -> sqrt(3)
    torch.testing.assert_close(out, _t([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]))


def test_value_loss_clipfrac_is_zero_when_clipped_and_unclipped_tie() -> None:
    _, stats = value_loss(_t([0.3]), _t([0.3]), _t([1.0]), _ones(1), clip_range=0.5)
    assert stats["vf_clipfrac"].item() == pytest.approx(0.0)


def test_validate_token_mask_keeps_float_dtype_and_floats_bool() -> None:
    assert validate_token_mask(torch.ones(1, 2, dtype=F64)).dtype == F64
    assert validate_token_mask(torch.ones(1, 2, dtype=torch.bool)).is_floating_point()


def test_entropy_from_logits_upcasts_half_logits_to_fp32() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(2, 3, 50, generator=gen) * 8).half()
    mask = torch.ones(2, 3)
    out = entropy_from_logits(logits, mask)
    assert out.dtype == torch.float32
    reference = torch.distributions.Categorical(logits=logits.double()).entropy().mean()
    torch.testing.assert_close(out, reference.float(), rtol=1e-5, atol=1e-5)
