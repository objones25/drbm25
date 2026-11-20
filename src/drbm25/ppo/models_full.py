"""A100 layout (spec §6.5): separate actor and critic, full fine-tune, FSDP2.

The actor (a causal LM) and the critic (a backbone without LM head plus a scalar value head)
hold fp32 master weights. Each is wrapped per decoder layer, then at the root, with
``fully_shard(..., mp_policy=MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=float32))``:
forwards all-gather the shards and run in ``compute_dtype`` (bf16 on A100), gradients are
reduce-scattered in fp32 and averaged over ranks during ``backward``. The trainer must therefore
not all-reduce gradients again. The value head is its own FSDP unit with an fp32 policy, so
values feed GAE in fp32. The frozen reference model is sharded the same way (no optimizer,
``requires_grad=False``) and kept in ``compute_dtype``.

Every forward is a collective (each FSDP unit all-gathers its parameters), so every rank must
run the same sequence of forwards. Generation must keep ranks in lockstep: vLLM (the A100
backend) samples from its own weight copy; HF ``generate`` needs ``synced_gpus=True``.

API facts checked against torch 2.10 (``inspect.signature`` and runs on a 2-rank gloo group):
  * ``fully_shard(module, *, mesh, reshard_after_forward, shard_placement_fn, mp_policy,
    offload_policy, ignored_params)``; ``MixedPrecisionPolicy(param_dtype, reduce_dtype,
    output_dtype, cast_forward_inputs=True)``. Parameters become ``DTensor`` shards (dim 0).
  * Without ``mesh`` the default mesh is built for the "current accelerator", which is MPS on an
    Apple-silicon Mac and fails (``torch.mps`` has no ``is_initialized``); the caller passes an
    explicit mesh of ``dist.device.type``.
  * ``clip_grad_norm_`` over ``DTensor`` parameters returns a replicated ``DTensor`` holding the
    global norm (``float()`` works); ``torch.linalg.vector_norm`` of a sharded ``DTensor`` is
    likewise global, so ``per_layer_grad_and_weight_norms`` and ``update_ratio`` need no change.
  * ``DTensor.full_tensor()`` all-gathers one tensor; ``torch.distributed.checkpoint.state_dict``
    ``get_state_dict`` / ``set_state_dict`` turn model + optimizer state into FQN-keyed,
    DCP-saveable dicts (``StateDictOptions(full_state_dict=False)``: sharded).
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from transformers import PreTrainedModel

ACTOR: Final = "actor"
CRITIC: Final = "critic"
_FULL: Final = StateDictOptions(full_state_dict=True)


class _Critic(nn.Module):
    """Backbone hidden states -> fp32 scalar value per position (``[B, L]``)."""

    def __init__(self, backbone: PreTrainedModel) -> None:
        super().__init__()
        self.backbone = backbone
        device = backbone.get_input_embeddings().weight.device
        self.value_head = nn.Linear(backbone.config.hidden_size, 1, device=device, dtype=torch.float32)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        # Autocast would run the fp32 head in bf16; the values feed GAE and must stay fp32.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.value_head(hidden.float()).squeeze(-1)


def _layers(model: nn.Module) -> nn.ModuleList:
    layers = model.layers if hasattr(model, "layers") else model.get_decoder().layers  # pyright: ignore[reportCallIssue]
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"expected decoder layers to be a ModuleList, got {type(layers).__name__}")
    return layers


def _shard(root: nn.Module, layers: nn.ModuleList, mesh: DeviceMesh, policy: MixedPrecisionPolicy) -> None:
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(root, mesh=mesh, mp_policy=policy)


def _gather(module: nn.Module, dtype: torch.dtype | None, device: torch.device | None) -> dict[str, Tensor]:
    """Full (unsharded) copies of ``module.state_dict()``; a collective, so call on every rank.

    Tied weights (embedding / LM head) are the same parameter object: gathered once, and the
    two names share the returned tensor.
    """
    out: dict[str, Tensor] = {}
    done: dict[int, Tensor] = {}
    for name, tensor in module.state_dict(keep_vars=True).items():
        if id(tensor) not in done:
            full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor.clone()
            done[id(tensor)] = full.detach().to(dtype=dtype or full.dtype, device=device or full.device)
        out[name] = done[id(tensor)]
    return out


class FullPolicyModels:
    """FSDP2-sharded actor, critic and frozen reference (implements ``PolicyModels``).

    ``actor`` and ``critic_backbone`` must hold fp32 weights (the master copy); ``ref`` is cast
    to ``compute_dtype``. ``critic_backbone`` is a model without an LM head (``AutoModel``).
    Modules may live on CPU: ``fully_shard`` moves each shard to the mesh's device. They are
    modified in place. Every method that touches weights is a collective: call it on every rank.
    """

    def __init__(
        self,
        actor: PreTrainedModel,
        critic_backbone: PreTrainedModel,
        ref: PreTrainedModel,
        *,
        mesh: DeviceMesh,
        compute_dtype: torch.dtype,
        gradient_checkpointing: bool,
    ) -> None:
        if critic_backbone.get_output_embeddings() is not None:
            raise TypeError(
                f"critic_backbone must be a backbone without an LM head (AutoModel), got {type(critic_backbone).__name__}"
            )
        for name, model in (("actor", actor), ("critic_backbone", critic_backbone)):
            dtypes = {p.dtype for p in model.parameters()}
            if dtypes != {torch.float32}:
                raise ValueError(f"{name} must hold fp32 master weights, got {sorted(map(str, dtypes))}")
        if gradient_checkpointing:
            # Non-reentrant: every layer input already requires grad (full FT), and it composes with FSDP2.
            for model in (actor, critic_backbone):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        policy = MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=torch.float32)
        critic = _Critic(critic_backbone)
        _shard(actor, _layers(actor), mesh, policy)
        fully_shard(
            critic.value_head,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
        )
        _shard(critic, _layers(critic_backbone), mesh, policy)
        nn.Module.to(ref, dtype=compute_dtype)  # transformers' decorated .to() confuses pyright
        ref.requires_grad_(False)
        ref.eval()
        _shard(ref, _layers(ref), mesh, MixedPrecisionPolicy())

        self._actor = actor
        self._critic = critic
        self._ref = ref
        self._compute_dtype = compute_dtype
        self._actor_params = list(actor.parameters())
        self._critic_params = list(critic.parameters())

    def _modules(self) -> dict[str, nn.Module]:
        return {ACTOR: self._actor, CRITIC: self._critic}

    # ------------------------------------------------------------------ forwards
    def actor_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Train mode iff grad: gradient checkpointing only runs in train mode, and Qwen2 has no dropout.
        self._actor.train(torch.is_grad_enabled())
        return self._actor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    def ref_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.no_grad():
            return self._ref(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    def critic_values(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        self._critic.train(torch.is_grad_enabled())
        return self._critic(input_ids, attention_mask)

    # ------------------------------------------------------------------ parameters
    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self._actor_params)

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self._critic_params)

    def generation_module(self) -> nn.Module:
        """The FSDP actor in eval mode; HF ``generate`` on it needs ``synced_gpus=True``."""
        self._actor.eval()
        return self._actor

    def decoder_layers(self) -> Sequence[nn.Module]:
        return list(_layers(self._actor))

    def discard_pending(self) -> None:
        """No-op: no adapters are switched, so no backward can read the wrong weights."""

    # ------------------------------------------------------------------ state
    def merged_actor_state_dict(self, *, device: torch.device | None = None) -> dict[str, Tensor]:
        """The full actor, in ``compute_dtype``, named as the plain HF model.

        A collective (one all-gather per tensor): call on every rank. The tensors land on
        ``device`` (default: the actor's device). On the actor's device the peak extra memory is
        one full actor copy in ``compute_dtype`` (6.2 GB at 3B bf16); with ``device=cpu`` the GPU
        holds one gathered tensor at a time and the copy is in host memory.
        """
        return _gather(self._actor, self._compute_dtype, device)

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Full fp32 copies of actor and critic on CPU, keyed ``actor.<name>`` / ``critic.<name>``.

        A collective that materialises both full models in host memory: for tests and small
        models. Training checkpoints use :meth:`sharded_state` with DCP instead.
        """
        return {
            f"{prefix}.{name}": tensor
            for prefix, module in self._modules().items()
            for name, tensor in _gather(module, None, torch.device("cpu")).items()
        }

    def load_trainable_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Inverse of :meth:`trainable_state_dict` (every rank passes the same full tensors)."""
        expected = {f"{p}.{n}" for p, m in self._modules().items() for n in m.state_dict(keep_vars=True)}
        missing, unexpected = sorted(expected - state.keys()), sorted(state.keys() - expected)
        if missing or unexpected:
            raise KeyError(f"trainable state mismatch: missing={missing} unexpected={unexpected}")
        for prefix, module in self._modules().items():
            own: dict[str, Any] = {
                name[len(prefix) + 1 :]: t for name, t in state.items() if name.startswith(prefix + ".")
            }
            set_model_state_dict(module, own, options=_FULL)

    def sharded_state(self, optimizers: Mapping[str, Optimizer]) -> dict[str, Any]:
        """DCP-saveable sharded state: ``{"actor"|"critic": {"model": ..., "optim": ...}}``.

        ``optimizers`` maps ``"actor"`` and ``"critic"`` to the optimizers over
        :meth:`actor_parameters` and :meth:`critic_parameters`. Pass the result to
        ``dcp.load`` to fill it in place, then to :meth:`load_sharded_state`.
        """
        self._check_optimizers(optimizers)
        out: dict[str, Any] = {}
        for name, module in self._modules().items():
            model_sd, optim_sd = get_state_dict(module, optimizers[name])
            out[name] = {"model": model_sd, "optim": optim_sd}
        return out

    def load_sharded_state(self, state: Mapping[str, Any], optimizers: Mapping[str, Optimizer]) -> None:
        self._check_optimizers(optimizers)
        for name, module in self._modules().items():
            set_state_dict(
                module,
                optimizers[name],
                model_state_dict=state[name]["model"],
                optim_state_dict=state[name]["optim"],
            )

    def _check_optimizers(self, optimizers: Mapping[str, Optimizer]) -> None:
        if set(optimizers) != {ACTOR, CRITIC}:
            raise KeyError(f"optimizers must be keyed {{'actor', 'critic'}}, got {sorted(optimizers)}")
