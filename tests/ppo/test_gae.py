import pytest
import torch

from drbm25.ppo.gae import gae, kl_penalized_rewards

F64 = torch.float64


def _t(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(list(rows), dtype=F64)


def test_kl_rewards_hand_values() -> None:
    out = kl_penalized_rewards(
        torch.tensor([1.0], dtype=F64),
        _t([-1.0, -1.0, -1.0]),
        _t([-1.5, -1.0, -0.5]),
        _t([1.0, 1.0, 1.0]),
        kl_coef=0.1,
    )
    torch.testing.assert_close(out, _t([-0.05, 0.0, 1.05]))


def test_kl_rewards_put_task_reward_on_last_valid_token() -> None:
    out = kl_penalized_rewards(
        torch.tensor([1.0], dtype=F64),
        _t([-1.0, -1.0, -1.0]),
        _t([-1.0, -1.0, -3.0]),
        _t([1.0, 1.0, 0.0]),
        kl_coef=0.1,
    )
    torch.testing.assert_close(out, _t([0.0, 1.0, 0.0]))


def test_kl_rewards_accept_bool_mask_and_per_row_rewards() -> None:
    out = kl_penalized_rewards(
        torch.tensor([1.0, -2.0], dtype=F64),
        torch.zeros(2, 3, dtype=F64),
        torch.zeros(2, 3, dtype=F64),
        torch.tensor([[True, True, True], [True, False, False]]),
        kl_coef=0.1,
    )
    torch.testing.assert_close(out, _t([0.0, 0.0, 1.0], [-2.0, 0.0, 0.0]))


def test_kl_rewards_reject_all_zero_mask_row() -> None:
    with pytest.raises(ValueError, match="row 1 has no valid tokens"):
        kl_penalized_rewards(
            torch.zeros(2, dtype=F64),
            torch.zeros(2, 2, dtype=F64),
            torch.zeros(2, 2, dtype=F64),
            _t([1.0, 1.0], [0.0, 0.0]),
            kl_coef=0.1,
        )


def test_kl_rewards_reject_shape_mismatch_naming_tensors() -> None:
    with pytest.raises(ValueError, match=r"logprobs_ref.*\(1, 2\)"):
        kl_penalized_rewards(
            torch.zeros(1, dtype=F64),
            torch.zeros(1, 3, dtype=F64),
            torch.zeros(1, 2, dtype=F64),
            torch.ones(1, 3, dtype=F64),
            kl_coef=0.1,
        )


def test_kl_rewards_reject_task_reward_wrong_length() -> None:
    with pytest.raises(ValueError, match=r"task_reward.*\(3,\)"):
        kl_penalized_rewards(
            torch.zeros(3, dtype=F64),
            torch.zeros(1, 2, dtype=F64),
            torch.zeros(1, 2, dtype=F64),
            torch.ones(1, 2, dtype=F64),
            kl_coef=0.1,
        )


def test_gae_gamma_lambda_one_hand_values() -> None:
    adv, ret = gae(
        _t([-0.05, 0.0, 1.05]),
        _t([0.2, 0.3, 0.4]),
        _t([1.0, 1.0, 1.0]),
        gamma=1.0,
        lam=1.0,
    )
    torch.testing.assert_close(adv, _t([0.8, 0.75, 0.65]))
    torch.testing.assert_close(ret, _t([1.0, 1.05, 1.05]))


def test_gae_discounting_hand_values() -> None:
    # r=[0,0,1], v=0, gamma=0.5, lam=0.5: delta=[0,0,1]; A_t = (gamma*lam)^(2-t)
    adv, ret = gae(
        _t([0.0, 0.0, 1.0]),
        _t([0.0, 0.0, 0.0]),
        _t([1.0, 1.0, 1.0]),
        gamma=0.5,
        lam=0.5,
    )
    torch.testing.assert_close(adv, _t([0.0625, 0.25, 1.0]))
    torch.testing.assert_close(ret, adv)


def test_gae_lambda_zero_is_one_step_td() -> None:
    adv, _ = gae(
        _t([1.0, 2.0, 3.0]),
        _t([0.5, 1.0, 1.5]),
        _t([1.0, 1.0, 1.0]),
        gamma=0.9,
        lam=0.0,
    )
    torch.testing.assert_close(adv, _t([1.0 + 0.9 * 1.0 - 0.5, 2.0 + 0.9 * 1.5 - 1.0, 3.0 - 1.5]))


def test_gae_zero_beyond_mask() -> None:
    kwargs = {"gamma": 1.0, "lam": 1.0}
    mask = _t([1.0, 1.0, 0.0])
    adv, ret = gae(_t([0.0, 1.0, 5.0]), _t([0.2, 0.3, 99.0]), mask, **kwargs)
    # Terminal at index 1: delta1 = 1 - 0.3, delta0 = 0.3 - 0.2; the 99 value must not leak.
    torch.testing.assert_close(adv, _t([0.8, 0.7, 0.0]))
    torch.testing.assert_close(ret, _t([1.0, 1.0, 0.0]))
    # Changing anything beyond the mask changes nothing inside it.
    adv2, ret2 = gae(_t([0.0, 1.0, -7.0]), _t([0.2, 0.3, -3.0]), mask, **kwargs)
    torch.testing.assert_close(adv2, adv)
    torch.testing.assert_close(ret2, ret)


def test_gae_rows_are_independent() -> None:
    rewards = _t([0.0, 0.0, 1.0], [0.0, 2.0, 0.0])
    values = _t([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    mask = _t([1.0, 1.0, 1.0], [1.0, 1.0, 0.0])
    adv, ret = gae(rewards, values, mask, gamma=0.9, lam=0.8)
    for i in range(2):
        a_i, r_i = gae(rewards[i : i + 1], values[i : i + 1], mask[i : i + 1], gamma=0.9, lam=0.8)
        torch.testing.assert_close(adv[i : i + 1], a_i)
        torch.testing.assert_close(ret[i : i + 1], r_i)


def test_gae_outputs_do_not_require_grad() -> None:
    values = torch.zeros(1, 2, dtype=F64, requires_grad=True)
    adv, ret = gae(torch.ones(1, 2, dtype=F64), values, torch.ones(1, 2, dtype=F64), gamma=1.0, lam=1.0)
    assert not adv.requires_grad
    assert not ret.requires_grad


def test_gae_rejects_shape_mismatch_naming_tensors() -> None:
    with pytest.raises(ValueError, match=r"values has shape \(1, 2\) but mask has shape \(1, 3\)"):
        gae(torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1, 3), gamma=1.0, lam=1.0)


@pytest.mark.parametrize(("gamma", "lam"), [(-0.1, 1.0), (1.1, 1.0), (1.0, -0.1), (1.0, 1.1)])
def test_gae_rejects_gamma_or_lambda_outside_unit_interval(gamma: float, lam: float) -> None:
    with pytest.raises(ValueError, match="gamma and lam must be in"):
        gae(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2), gamma=gamma, lam=lam)


def test_gae_rejects_mask_with_values_other_than_zero_one() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        gae(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[1.0, 0.5]]), gamma=1.0, lam=1.0)


def test_gae_gamma_zero_is_reward_minus_value() -> None:
    adv, _ = gae(_t([1.0, 2.0]), _t([0.5, 1.0]), _t([1.0, 1.0]), gamma=0.0, lam=1.0)
    torch.testing.assert_close(adv, _t([0.5, 1.0]))
