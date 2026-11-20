"""Rollout engine protocol and the batch it returns."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

from drbm25.ppo.models import PolicyModels


@dataclass(frozen=True)
class RolloutBatch:
    """One generation batch of ``B`` prompts.

    ``prompt_ids``/``prompt_mask`` are ``[B, P]``, left-padded. ``response_ids`` and
    ``response_mask`` are ``[B, R]``, right-padded with the pad id; the mask covers the
    generated tokens up to and including the first EOS, so it is 0 for everything after.
    ``hit_max_tokens[b]`` is True iff row ``b`` produced no EOS within ``max_new_tokens``.
    """

    prompt_ids: Tensor
    prompt_mask: Tensor
    response_ids: Tensor
    response_mask: Tensor
    response_text: tuple[str, ...]
    hit_max_tokens: Tensor


@runtime_checkable
class RolloutEngine(Protocol):
    def generate(self, prompts: Sequence[str], *, greedy: bool) -> RolloutBatch: ...

    def sync_weights(self, models: PolicyModels) -> None:
        """Make the engine sample from ``models``' current actor weights."""
        ...
