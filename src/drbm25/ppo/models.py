"""Policy models for PPO: the ``PolicyModels`` protocol and its LoRA implementation.

V100 layout (spec §6.5): one frozen base causal LM shared by two LoRA adapters,
``"actor"`` and ``"critic"``, plus an fp32 value head. The reference policy is the base
with adapters disabled, so no second copy of the weights is kept.

API facts checked against the installed versions (peft 0.21.2, transformers 4.57.6):
  * ``get_peft_model(..., autocast_adapter_dtype=True)`` (the default) upcasts fp16/bf16
    adapter weights to fp32, so the adapters train in fp32 over an fp16 base.
  * ``BaseTunerLayer.set_adapter`` sets ``requires_grad=False`` on every inactive adapter;
    ``add_adapter`` leaves the new adapter frozen. ``_activate`` re-enables both adapters
    after each switch, so "trainable" does not depend on which adapter ran last.
  * transformers' ``gradient_checkpointing_enable`` defaults to ``use_reentrant=True``;
    with a frozen base no checkpoint input requires grad and the LoRA grads come back
    None (torch warns "None of the inputs have requires_grad=True"), so
    ``enable_input_require_grads()`` is needed. Checkpointing only runs in train mode
    (``GradientCheckpointingLayer.__call__`` checks ``self.training``).
  * ``output_hidden_states=True``: the last entry is the post-final-norm hidden state
    (equal to ``model.model(...).last_hidden_state``).
"""

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.layer import LoraLayer
from torch import Tensor, nn
from transformers import PreTrainedModel

from drbm25.config import LoraCfg

ACTOR = "actor"
CRITIC = "critic"
_HEAD_PREFIX = "value_head."
_Forward = Callable[[], Tensor]


class AdapterSwitchError(RuntimeError):
    """A grad-enabled forward through one adapter while another adapter's backward is pending.

    Under reentrant gradient checkpointing the LoRA layers are re-run during backward and
    read the adapter that is active *then*; running a different adapter in between would
    make the pending backward silently use the wrong weights.
    """


def token_logprobs(logits: Tensor, input_ids: Tensor) -> Tensor:
    """``[B, L-1]`` fp32: position ``t`` is ``log p(input_ids[:, t+1] | input_ids[:, :t+1])``.

    Logits are upcast to fp32 before ``log_softmax``: fp16 log-probs lose the precision
    the PPO ratio ``exp(new - old)`` depends on.
    """
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError(
            f"token_logprobs needs logits [B, L, V] and input_ids [B, L] with L >= 2, "
            f"got {tuple(logits.shape)} and {tuple(input_ids.shape)}"
        )
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


@runtime_checkable
class PolicyModels(Protocol):
    def actor_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: ...

    def ref_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Reference-policy logits, computed without grad."""
        ...

    def critic_values(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """``[B, L]`` fp32 value of every position."""
        ...

    def actor_parameters(self) -> list[nn.Parameter]: ...

    def critic_parameters(self) -> list[nn.Parameter]: ...

    def generation_module(self) -> nn.Module:
        """A module whose forward/generate run the actor."""
        ...

    def trainable_state_dict(self) -> dict[str, Tensor]: ...

    def load_trainable_state_dict(self, state: Mapping[str, Tensor]) -> None: ...

    def decoder_layers(self) -> Sequence[nn.Module]: ...

    def discard_pending(self) -> None:
        """Clear ALL pending marks; only for outputs that will never be backpropagated (see ``AdapterSwitchError``)."""
        ...

    def merged_actor_state_dict(self) -> dict[str, Tensor]:
        """The actor as plain base-model weights, named as the base's own ``state_dict()``."""
        ...


class LoraPolicyModels:
    """Actor and critic LoRA adapters on one frozen base, plus a zero-initialised fp32 value head.

    ``base`` is modified in place (LoRA layers are injected into it) and must already be
    on its final device. Each forward sets train mode iff grad is enabled: checkpointing
    only runs in train mode, and a reentrant checkpoint under ``no_grad`` warns that no
    input requires grad. So LoRA dropout (``lora.dropout``) is active only in grad-enabled
    forwards; keep it 0 when old (no-grad) and new log-probs must agree at ratio 1.

    With ``gradient_checkpointing=True`` every grad-enabled actor/critic forward stays
    *pending* until backward reaches its output (a tensor hook clears it). While one is
    pending, a grad-enabled forward through the other adapter raises ``AdapterSwitchError``
    (backward each loss before running the other adapter), and no-grad forwards (ref,
    rollout values) restore the pending adapter and train mode afterwards. Call
    ``discard_pending()`` when an output is intentionally never backpropagated.
    """

    def __init__(self, base: PreTrainedModel, lora: LoraCfg, *, gradient_checkpointing: bool) -> None:
        config = LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base, config, adapter_name=ACTOR)
        if not isinstance(peft_model, PeftModel):
            raise TypeError(f"get_peft_model returned {type(peft_model).__name__}, expected PeftModel")
        peft_model.add_adapter(CRITIC, config)
        if gradient_checkpointing:
            # Enabled after wrapping so PeftModel.__init__ does not also register the input hook.
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            base.enable_input_require_grads()

        self._base = base
        self._peft = peft_model
        self._checkpointing = gradient_checkpointing
        self._pending_adapter: str | None = None
        self._pending_tokens: set[int] = set()
        self._active_adapter = ACTOR
        self._tokens = itertools.count()
        device = base.get_input_embeddings().weight.device
        self._value_head = nn.Linear(base.config.hidden_size, 1, device=device, dtype=torch.float32)
        nn.init.zeros_(self._value_head.weight)
        nn.init.zeros_(self._value_head.bias)

        # Collected after add_adapter: autocast_adapter_dtype replaces adapter Parameters.
        self._lora_linears = {name: m for name, m in base.named_modules() if isinstance(m, LoraLayer)}
        for name, module in self._lora_linears.items():
            if not isinstance(module, LoraLinear):
                raise TypeError(f"{name}: only LoRA Linear layers are supported, got {type(module).__name__}")
        self._actor = self._adapter_params(ACTOR)
        self._critic_lora = self._adapter_params(CRITIC)
        if not self._actor or len(self._actor) != len(self._critic_lora):
            raise ValueError(f"no LoRA layers matched target_modules={lora.target_modules}")
        self._activate(ACTOR, training=False)

    def _adapter_params(self, adapter: str) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self._lora_linears.values():
            for table in (module.lora_A, module.lora_B):
                params.extend(table[adapter].parameters())
        return params

    def _activate(self, adapter: str, *, training: bool) -> None:
        self._peft.train(training)
        self._peft.set_adapter(adapter)
        self._active_adapter = adapter
        for p in self._actor + self._critic_lora:
            p.requires_grad_(True)

    @contextmanager
    def _no_grad_detour(self) -> Iterator[None]:
        """Undo a no-grad forward's side effects on the adapter state, in every mode.

        Restores the adapter and train mode that were active before (or, if a checkpointed
        backward is pending, its adapter in train mode), and ``requires_grad=True`` on both
        adapters: peft's ``set_adapter`` -- also called by ``disable_adapter()`` on exit --
        freezes the inactive adapter, and autograd skips accumulating into a parameter whose
        ``requires_grad`` is False at backward time, so an earlier forward's grads would
        silently come back None.
        """
        adapter, training = self._active_adapter, self._peft.training
        try:
            yield
        finally:
            if self._pending_adapter is not None:
                adapter, training = self._pending_adapter, True
            self._activate(adapter, training=training)

    def _run(self, adapter: str, forward_output: _Forward) -> Tensor:
        grad = torch.is_grad_enabled()
        pending = self._pending_adapter
        if grad and pending is not None and pending != adapter:
            raise AdapterSwitchError(
                f"backward the {pending} loss before running {adapter} with gradient checkpointing "
                f"(the pending {pending} backward re-runs its layers with whichever adapter is active); "
                f"call discard_pending() if the {pending} output will never be backpropagated"
            )
        if grad:
            self._activate(adapter, training=True)
            out = forward_output()
        else:
            with self._no_grad_detour():
                self._activate(adapter, training=False)
                out = forward_output()
        if grad and self._checkpointing and out.requires_grad:
            self._mark_pending(adapter, out)
        return out

    def _mark_pending(self, adapter: str, out: Tensor) -> None:
        token = next(self._tokens)
        self._pending_tokens.add(token)
        self._pending_adapter = adapter

        # Fires on ANY gradient reaching `out` -- including torch.autograd.grad and a first
        # backward(retain_graph=True) -- so the mark can clear before a later backward of the same
        # graph; never interleave a different-adapter forward between such calls.
        def clear(_grad: Tensor) -> None:
            self._pending_tokens.discard(token)
            if not self._pending_tokens:
                self._pending_adapter = None

        out.register_hook(clear)

    def discard_pending(self) -> None:
        """Clear ALL pending marks (every adapter, every outstanding output).

        Use only when the pending outputs will never be backpropagated; backpropagating one
        afterwards across a different-adapter forward gives silently wrong grads.
        """
        self._pending_tokens.clear()
        self._pending_adapter = None

    def actor_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        def forward() -> Tensor:
            return self._peft(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

        return self._run(ACTOR, forward)

    def ref_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.no_grad(), self._no_grad_detour(), self._peft.disable_adapter():
            self._peft.eval()
            out = self._peft(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits

    def critic_values(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        def forward() -> Tensor:
            # logits_to_keep=1: the LM head is not needed, so skip the [B, L, V] projection.
            out = self._peft(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                logits_to_keep=1,
                use_cache=False,
            )
            return self._value_head(out.hidden_states[-1].float()).squeeze(-1)

        return self._run(CRITIC, forward)

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self._actor)

    def critic_parameters(self) -> list[nn.Parameter]:
        return [*self._critic_lora, *self._value_head.parameters()]

    def generation_module(self) -> nn.Module:
        """The PEFT model with the actor active, in eval mode (call under ``torch.inference_mode()``).

        The returned module is shared: its active adapter changes whenever ``critic_values``
        (or ``actor_logits``) runs, so call ``generation_module()`` again right before each
        generate. Refused with ``AdapterSwitchError`` while a checkpointed backward is pending.
        """
        if self._pending_adapter is not None:
            raise AdapterSwitchError(
                f"backward the {self._pending_adapter} loss (or discard_pending()) before generation"
            )
        self._activate(ACTOR, training=False)
        return self._peft

    def _named_trainable(self) -> dict[str, nn.Parameter]:
        trainable = {id(p) for p in self._actor + self._critic_lora}
        named = {name: p for name, p in self._base.named_parameters() if id(p) in trainable}
        if len(named) != len(trainable):
            raise RuntimeError("some LoRA parameters are not registered on the base model")
        named.update({_HEAD_PREFIX + name: p for name, p in self._value_head.named_parameters()})
        return named

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Detached copies of both adapters and the value head (base weights excluded)."""
        return {name: p.detach().clone() for name, p in self._named_trainable().items()}

    def load_trainable_state_dict(self, state: Mapping[str, Tensor]) -> None:
        params = self._named_trainable()
        missing = sorted(params.keys() - state.keys())
        unexpected = sorted(state.keys() - params.keys())
        if missing or unexpected:
            raise KeyError(f"trainable state mismatch: missing={missing} unexpected={unexpected}")
        for name, p in params.items():
            if state[name].shape != p.shape:
                raise ValueError(f"{name}: expected shape {tuple(p.shape)}, got {tuple(state[name].shape)}")
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(state[name])

    def decoder_layers(self) -> Sequence[nn.Module]:
        layers = self._base.get_decoder().layers
        if not isinstance(layers, nn.ModuleList):
            raise TypeError(f"expected decoder.layers to be a ModuleList, got {type(layers).__name__}")
        return list(layers)

    def merged_actor_state_dict(self) -> dict[str, Tensor]:
        """Base weights with the actor merged (``W + scale * B @ A``), keyed as the plain base model.

        The live model is not modified (no ``merge_and_unload``). Merged weights are new
        tensors; every other entry shares storage with the live model and must be
        treated as read-only.
        """
        merged: dict[str, Tensor] = {}
        for key, tensor in self._base.state_dict().items():
            if ".lora_" in key:
                continue
            merged[key.replace(".base_layer.", ".")] = tensor.detach()
        with torch.no_grad():
            for name, module in self._lora_linears.items():
                if not isinstance(module, LoraLinear):
                    raise TypeError(f"{name}: only LoRA Linear layers are supported")
                key = f"{name}.weight"
                weight = merged[key]
                delta = module.get_delta_weight(ACTOR)
                merged[key] = (weight.float() + delta.float()).to(weight.dtype)
        return merged
