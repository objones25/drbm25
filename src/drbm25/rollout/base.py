"""Rollout engine protocol, the batch it returns, and the prompt/response handling both engines share."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

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


@dataclass(frozen=True)
class SpecialIds:
    pad: int
    eos: int


def special_ids(tokenizer: PreTrainedTokenizerBase) -> SpecialIds:
    """Pad and EOS ids, refusing tokenizers whose responses could not be masked or batched."""
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    if not isinstance(pad_id, int) or not isinstance(eos_id, int):
        raise ValueError(f"tokenizer needs integer pad_token_id and eos_token_id, got {pad_id!r} and {eos_id!r}")
    if pad_id == eos_id:
        raise ValueError("tokenizer pad_token_id equals eos_token_id; the response mask cannot tell padding from EOS")
    if tokenizer.padding_side != "left":
        raise ValueError(
            f"tokenizer padding_side must be 'left' for batched generation, got {tokenizer.padding_side!r}"
        )
    return SpecialIds(pad=pad_id, eos=eos_id)


def encode_prompts(
    tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str], device: torch.device, *, max_prompt_tokens: int
) -> tuple[Tensor, Tensor]:
    """Left-padded ``[B, P]`` prompt ids and attention mask; refuses an empty list or an over-long prompt."""
    if len(prompts) == 0:
        raise ValueError("prompts must be non-empty")
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True)
    prompt_ids: Tensor = enc.data["input_ids"].to(device)
    prompt_mask: Tensor = enc.data["attention_mask"].to(device)
    longest = int(prompt_mask.sum(dim=1).max().item())
    if longest > max_prompt_tokens:
        raise ValueError(f"prompt has {longest} tokens, over max_prompt_tokens={max_prompt_tokens}")
    return prompt_ids, prompt_mask


def right_pad(rows: Sequence[Tensor], pad_id: int) -> Tensor:
    """Stack ``[n_i, R_i]`` chunks (or ``[R_i]`` rows) into one right-padded ``[sum n_i, max R_i]`` tensor."""
    chunks = [r.unsqueeze(0) if r.ndim == 1 else r for r in rows]
    width = max(c.shape[1] for c in chunks)
    padded = [torch.nn.functional.pad(c, (0, width - c.shape[1]), value=pad_id) for c in chunks]
    return torch.cat(padded, dim=0)


def finalize_rollout(
    prompt_ids: Tensor,
    prompt_mask: Tensor,
    response_ids: Tensor,
    *,
    tokenizer: PreTrainedTokenizerBase,
    ids: SpecialIds,
) -> RolloutBatch:
    """Cut each right-padded response after its first EOS, build the mask, decode the text."""
    if response_ids.ndim != 2 or response_ids.shape[0] != prompt_ids.shape[0]:
        raise ValueError(
            f"response_ids must be [B, R] with B = {prompt_ids.shape[0]} prompts, got {tuple(response_ids.shape)}"
        )
    is_eos = (response_ids == ids.eos).long()
    # Keep tokens up to and including the first EOS (cumsum is 1 there, minus itself is 0).
    keep = (is_eos.cumsum(dim=1) - is_eos) == 0
    response_ids = torch.where(keep, response_ids, torch.full_like(response_ids, ids.pad))
    texts = tuple(
        tokenizer.decode(row[m].tolist(), skip_special_tokens=True) for row, m in zip(response_ids, keep, strict=True)
    )
    return RolloutBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        response_ids=response_ids,
        response_mask=keep.long(),
        response_text=texts,
        hit_max_tokens=~(is_eos.sum(dim=1) > 0),
    )
