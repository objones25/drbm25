import math

import pytest
import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM

from drbm25.obs.metrics import (
    ActivationProbe,
    explained_variance,
    grad_norm,
    per_layer_grad_and_weight_norms,
    qk_logit_bound,
    snapshot,
    update_ratio,
)
from drbm25.tiny import build_tiny_qwen2


def _tiny() -> Qwen2ForCausalLM:
    model = build_tiny_qwen2()
    assert isinstance(model, Qwen2ForCausalLM)
    return model


def _layers(model: Qwen2ForCausalLM) -> list[Qwen2DecoderLayer]:
    layers = [layer for layer in model.model.layers if isinstance(layer, Qwen2DecoderLayer)]
    assert len(layers) == len(model.model.layers)
    return layers


def _backward_tiny(model: Qwen2ForCausalLM) -> torch.Tensor:
    ids = torch.arange(12).reshape(2, 6)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    return ids


def test_grad_norm_hand_value():
    a = nn.Parameter(torch.zeros(1))
    b = nn.Parameter(torch.zeros(1))
    a.grad = torch.tensor([3.0])
    b.grad = torch.tensor([4.0])
    assert grad_norm([a, b]) == pytest.approx(5.0)


def test_grad_norm_without_any_grad_raises():
    with pytest.raises(ValueError, match="no gradients"):
        grad_norm([nn.Parameter(torch.zeros(1))])


def test_per_layer_keys_and_values():
    model = _tiny()
    _backward_tiny(model)
    layers = _layers(model)
    out = per_layer_grad_and_weight_norms(layers, "actor")
    expected_keys = {
        f"actor/layer_{i:02d}/{part}/{kind}"
        for i in range(2)
        for part in ("attn", "mlp")
        for kind in ("grad_norm", "weight_norm")
    }
    assert set(out) == expected_keys
    assert all(type(v) is float for v in out.values())
    mlp_params = list(layers[1].mlp.parameters())
    want_w = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in mlp_params))
    want_g = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in mlp_params if p.grad is not None))
    assert out["actor/layer_01/mlp/weight_norm"] == pytest.approx(want_w, rel=1e-5)
    assert out["actor/layer_01/mlp/grad_norm"] == pytest.approx(want_g, rel=1e-5)


def test_per_layer_covers_only_trainable_params_and_omits_missing_grads():
    model = _tiny()
    layers = _layers(model)
    for p in layers[0].self_attn.parameters():
        p.requires_grad_(False)
    out = per_layer_grad_and_weight_norms(layers, "p")
    assert "p/layer_00/attn/weight_norm" not in out  # nothing trainable
    assert "p/layer_00/mlp/weight_norm" in out
    assert "p/layer_00/mlp/grad_norm" not in out  # no backward yet: omit, not 0
    _backward_tiny(model)
    out = per_layer_grad_and_weight_norms(layers, "p")
    assert "p/layer_00/mlp/grad_norm" in out
    assert "p/layer_00/attn/grad_norm" not in out


def test_activation_probe_reports_stats_for_each_layer():
    model = _tiny()
    layers = _layers(model)
    with ActivationProbe(layers, "act") as probe:
        model(input_ids=torch.arange(6).reshape(1, 6))
        stats = probe.stats()
    assert set(stats) == {
        f"act/layer_{i:02d}/{k}" for i in range(2) for k in ("mean", "std", "absmax", "nonfinite_frac")
    }
    assert all(type(v) is float for v in stats.values())
    assert stats["act/layer_00/nonfinite_frac"] == pytest.approx(0.0)
    assert stats["act/layer_00/absmax"] > 0.0


def test_activation_probe_captures_only_first_forward():
    model = _tiny()
    layers = _layers(model)
    with ActivationProbe(layers, "a") as probe:
        model(input_ids=torch.arange(6).reshape(1, 6))
        first = probe.stats()
        model(input_ids=torch.arange(6, 12).reshape(1, 6))
        assert probe.stats() == first


def test_activation_probe_stats_before_forward_raises():
    layers = _layers(_tiny())
    with ActivationProbe(layers, "a") as probe, pytest.raises(RuntimeError, match="no forward"):
        probe.stats()


def test_activation_probe_removes_hooks_on_exit():
    model = _tiny()
    layers = _layers(model)
    with ActivationProbe(layers, "a"):
        assert all(len(layer._forward_hooks) == 1 for layer in layers)
    assert all(len(layer._forward_hooks) == 0 for layer in layers)


def test_activation_probe_removes_hooks_when_body_raises():
    layers = _layers(_tiny())
    with pytest.raises(KeyError), ActivationProbe(layers, "a"):
        raise KeyError("boom")
    assert all(len(layer._forward_hooks) == 0 for layer in layers)


def test_activation_probe_counts_nonfinite():
    model = _tiny()
    layers = _layers(model)

    def inject_inf(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        output = output.clone()
        output[0, 0, 0] = float("inf")
        return output

    handle = layers[0].register_forward_hook(inject_inf)  # runs before the probe's hook
    try:
        with ActivationProbe(layers[:1], "a") as probe:
            layers[0](torch.randn(1, 4, 32), position_embeddings=_rope(model, 4))
            stats = probe.stats()
    finally:
        handle.remove()
    assert stats["a/layer_00/nonfinite_frac"] == pytest.approx(1 / (4 * 32))
    assert math.isfinite(stats["a/layer_00/mean"])  # moments use finite entries only


def test_update_ratio_hand_value():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    before = snapshot([p])
    with torch.no_grad():
        p.copy_(torch.tensor([3.0, 4.5]))
    assert update_ratio(before, [p]) == pytest.approx(0.1)


def test_snapshot_is_a_copy():
    p = nn.Parameter(torch.tensor([1.0]))
    snap = snapshot([p])
    with torch.no_grad():
        p.add_(1.0)
    assert snap[0].item() == pytest.approx(1.0)


def test_update_ratio_rejects_length_mismatch_and_zero_norm():
    p = nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="length"):
        update_ratio([], [p])
    with pytest.raises(ValueError, match="zero"):
        update_ratio([torch.zeros(1)], [p])


def _rope(model: Qwen2ForCausalLM, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(1, seq, 32)
    pos = torch.arange(seq).unsqueeze(0)
    return model.model.rotary_emb(x, pos)


def _true_max_logit(model: Qwen2ForCausalLM, layer: Qwen2DecoderLayer, hidden: torch.Tensor) -> float:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    attn = layer.self_attn
    b, t, _ = hidden.shape
    x = layer.input_layernorm(hidden)
    q = attn.q_proj(x).view(b, t, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(b, t, -1, attn.head_dim).transpose(1, 2)
    cos, sin = _rope(model, t)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
    logits = (q @ k.transpose(-1, -2)) / attn.head_dim**0.5
    return float(logits.abs().max())


@pytest.mark.parametrize("layer_index", [0, 1])
def test_qk_bound_is_upper_bound_of_true_logits(layer_index: int):
    torch.manual_seed(0)
    model = _tiny()
    layer = _layers(model)[layer_index]
    hidden = torch.randn(2, 8, 32) * 3
    with torch.no_grad():
        true = _true_max_logit(model, layer, hidden)
        bound = qk_logit_bound(layer, hidden)
    assert type(bound) is float
    assert bound >= true
    assert bound < 50 * true + 1  # not vacuous


def test_qk_bound_rejects_non_3d_hidden():
    layer = _tiny().model.layers[0]
    with pytest.raises(ValueError, match=r"\[B, T, H\]"):
        qk_logit_bound(layer, torch.randn(8, 32))


def test_explained_variance_perfect_is_one_and_constant_is_zero():
    returns = torch.tensor([[1.0, 2.0, 3.0, 99.0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    assert explained_variance(returns.clone(), returns, mask) == pytest.approx(1.0)
    mean_pred = torch.full_like(returns, 2.0)  # masked mean of returns
    assert explained_variance(mean_pred, returns, mask) == pytest.approx(0.0, abs=1e-7)


def test_explained_variance_zero_variance_returns_is_zero_not_nan():
    returns = torch.full((1, 3), 5.0)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    assert explained_variance(values, returns, torch.ones(1, 3)) == pytest.approx(0.0)


def test_explained_variance_rejects_empty_mask():
    with pytest.raises(ValueError, match="no valid tokens"):
        explained_variance(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2))
