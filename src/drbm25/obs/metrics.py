"""Model and training health metrics for the PPO trainer.

The trainer calls these every ``deep_log_every`` steps, so hooks and extra forwards are
paid only on those steps. Every returned metric is a plain Python ``float`` ready for
W&B. Names follow ``f"{prefix}/layer_{i:02d}/..."``.

Reading the metrics (transformer-architecture ``stability.md``): a growing final-layer
activation norm or a growing ``qk_logit_bound`` precedes loss spikes; ``update_ratio``
near 1e-3 is a healthy step size for Adam-style optimisers.
"""

import math
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Self

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from drbm25.ppo.losses import validate_token_mask

_SUBMODULES = ("attn", "mlp")
_ATTR_FOR_SUBMODULE = {"attn": "self_attn", "mlp": "mlp"}


def grad_norm(params: Sequence[nn.Parameter]) -> float:
    """Global L2 norm of the gradients of ``params`` (call before clipping).

    Parameters whose ``.grad`` is ``None`` are skipped; if none has a gradient the call
    is a programmer error (metric requested before ``backward``) and raises ``ValueError``.
    """
    grads = [p.grad.detach().float() for p in params if p.grad is not None]
    if not grads:
        raise ValueError("no gradients: call grad_norm after backward()")
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))


def _l2(tensors: Sequence[Tensor]) -> float:
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t.detach().float()) for t in tensors])))


def per_layer_grad_and_weight_norms(layers: Sequence[nn.Module], prefix: str) -> dict[str, float]:
    """L2 norms of the TRAINABLE parameters under each layer's ``self_attn`` and ``mlp``.

    With LoRA these are the LoRA A/B matrices; with full fine-tuning, the full weights.
    ``weight_norm`` is emitted when the submodule has any trainable parameter;
    ``grad_norm`` only when at least one of them has a gradient (otherwise the key is
    omitted rather than reported as 0). Layernorm weights sit outside these submodules
    and are not covered.
    """
    out: dict[str, float] = {}
    for i, layer in enumerate(layers):
        for name in _SUBMODULES:
            module = getattr(layer, _ATTR_FOR_SUBMODULE[name])
            trainable = [p for p in module.parameters() if p.requires_grad]
            if not trainable:
                continue
            key = f"{prefix}/layer_{i:02d}/{name}"
            out[f"{key}/weight_norm"] = _l2(trainable)
            grads = [p.grad for p in trainable if p.grad is not None]
            if grads:
                out[f"{key}/grad_norm"] = _l2(grads)
    return out


class ActivationProbe:
    """Context manager capturing output-activation stats of each decoder layer.

    Records the layer output (first element if the layer returns a tuple) of the FIRST
    forward after entering, in fp32 under no-grad; later forwards are ignored. Padding
    positions are included. ``mean``/``std``/``absmax`` use finite entries only (NaN if
    there are none); ``nonfinite_frac`` is the share of NaN/inf entries. All hooks are
    removed on exit, including when the body raises.
    """

    def __init__(self, layers: Sequence[nn.Module], prefix: str) -> None:
        self._layers = list(layers)
        self._prefix = prefix
        self._handles: list[RemovableHandle] = []
        self._stats: dict[str, float] = {}

    def _make_hook(self, index: int) -> Callable[[nn.Module, tuple, Tensor | tuple], None]:
        key = f"{self._prefix}/layer_{index:02d}"

        def hook(_module: nn.Module, _inputs: tuple, output: Tensor | tuple) -> None:
            if f"{key}/mean" in self._stats:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            self._stats.update(_activation_stats(hidden, key))

        return hook

    def __enter__(self) -> Self:
        try:
            for i, layer in enumerate(self._layers):
                self._handles.append(layer.register_forward_hook(self._make_hook(i)))
        except BaseException:
            self._remove()
            raise
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._remove()

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def stats(self) -> dict[str, float]:
        if not self._stats:
            raise RuntimeError("no forward pass captured yet: run the model inside the with-block")
        return dict(self._stats)


@torch.no_grad()
def _activation_stats(hidden: Tensor, key: str) -> dict[str, float]:
    x = hidden.detach().float()
    finite = torch.isfinite(x)
    values = x[finite]
    nan = float("nan")
    return {
        f"{key}/mean": float(values.mean()) if values.numel() else nan,
        f"{key}/std": float(values.std(correction=0)) if values.numel() else nan,
        f"{key}/absmax": float(values.abs().max()) if values.numel() else nan,
        f"{key}/nonfinite_frac": float((~finite).sum()) / x.numel(),
    }


def snapshot(params: Sequence[nn.Parameter]) -> list[Tensor]:
    """Detached copies of ``params`` to pass to :func:`update_ratio` after the optimizer step."""
    return [p.detach().clone() for p in params]


def update_ratio(before: Sequence[Tensor], params: Sequence[nn.Parameter]) -> float:
    """``||theta_after - theta_before|| / ||theta_before||`` over all ``params`` (global L2)."""
    if len(before) != len(params):
        raise ValueError(f"length mismatch: {len(before)} snapshots vs {len(params)} params")
    delta = _l2([p.detach() - b for b, p in zip(before, params, strict=True)])
    base = _l2(before)
    if base == 0.0:
        raise ValueError("snapshot has zero norm: update ratio is undefined")
    return delta / base


@torch.no_grad()
def qk_logit_bound(layer: nn.Module, hidden: Tensor) -> float:
    """Upper bound on ``|q.k| / sqrt(head_dim)`` of a Qwen2-style decoder layer's attention.

    ``hidden`` is ``[B, T, H]``, the INPUT to the decoder layer: ``input_layernorm`` is
    applied before ``q_proj``/``k_proj`` as in the real forward. Per query head the bound
    is ``max_i ||q_i|| * max_j ||k_j|| / sqrt(head_dim)`` with ``k`` from the head's GQA
    group; the result is the max over heads. RoPE is a rotation, so it preserves norms and
    the bound holds with it applied (Cauchy-Schwarz). Values above 1e4 flag attention
    logit growth.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, T, H], got shape {tuple(hidden.shape)}")
    if not isinstance(layer, Qwen2DecoderLayer):
        raise TypeError(f"expected a Qwen2DecoderLayer, got {type(layer).__name__}")
    attn = layer.self_attn
    head_dim: int = attn.head_dim
    groups: int = attn.num_key_value_groups
    x = layer.input_layernorm(hidden)
    b, t, _ = x.shape
    q = attn.q_proj(x).view(b, t, -1, head_dim).float()
    k = attn.k_proj(x).view(b, t, -1, head_dim).float()
    q_max = torch.linalg.vector_norm(q, dim=-1).amax(dim=(0, 1))  # [n_heads]
    k_max = torch.linalg.vector_norm(k, dim=-1).amax(dim=(0, 1))  # [n_kv_heads]
    bound = q_max * k_max.repeat_interleave(groups) / math.sqrt(head_dim)
    return float(bound.max())


def explained_variance(values: Tensor, returns: Tensor, mask: Tensor) -> float:
    """``1 - Var(returns - values) / Var(returns)`` over masked tokens.

    1.0 is a perfect critic, 0.0 no better than predicting the mean return, negative
    worse. When ``Var(returns) == 0`` the ratio is undefined; this returns 0.0 (not NaN)
    so the W&B series stays plottable.
    """
    mask_f = validate_token_mask(mask, values=values, returns=returns)
    selected = mask_f.bool()
    if not bool(selected.any()):
        raise ValueError("mask has no valid tokens")
    v = values.detach().float()[selected]
    r = returns.detach().float()[selected]
    var_returns = float(r.var(correction=0))
    if var_returns == 0.0:
        return 0.0
    return 1.0 - float((r - v).var(correction=0)) / var_returns
