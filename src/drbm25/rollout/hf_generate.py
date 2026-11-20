"""Rollout engine backed by ``transformers`` ``generate`` on the live actor.

API facts checked against transformers 4.57.6 (inspect of ``GenerationMixin.generate``
and ``_get_logits_processor``):
  * ``generate(**kwargs)`` overrides ``GenerationConfig`` fields; ``logits_processor``
    takes a ``LogitsProcessorList`` that is merged with the built-in ones.
  * ``top_k=0`` and ``top_p=1.0`` add no warper. ``GenerationConfig`` defaults to
    ``top_k=50`` and an instruct checkpoint's ``generation_config.json`` can set
    ``top_k`` / ``repetition_penalty`` too, so every sampling field is passed explicitly.
  * With ``do_sample=False`` a ``temperature != 1.0`` triggers a validation warning, so
    greedy passes ``temperature=1.0``.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase

from drbm25.config import RolloutCfg
from drbm25.ppo.models import PolicyModels
from drbm25.rollout.base import RolloutBatch


class HFGenerateEngine:
    """Samples from the live PEFT actor; ``sync_weights`` is a no-op (same module).

    Each ``generate`` call fetches ``models.generation_module()`` afresh (it activates the
    actor in eval mode and refuses while a checkpointed backward is pending), so there is
    nothing to restore afterwards. It runs under ``torch.no_grad`` rather than
    ``inference_mode`` so the returned tensors can feed grad-enabled forwards later.
    ``extra_logits_processors`` exists for tests (forcing or banning tokens).
    """

    def __init__(
        self,
        models: PolicyModels,
        tokenizer: PreTrainedTokenizerBase,
        cfg: RolloutCfg,
        device: torch.device,
        *,
        extra_logits_processors: Sequence[LogitsProcessor] = (),
    ) -> None:
        pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        if not isinstance(pad_id, int) or not isinstance(eos_id, int):
            raise ValueError(f"tokenizer needs integer pad_token_id and eos_token_id, got {pad_id!r} and {eos_id!r}")
        if pad_id == eos_id:
            raise ValueError(
                "tokenizer pad_token_id equals eos_token_id; the response mask cannot tell padding from EOS"
            )
        if tokenizer.padding_side != "left":
            raise ValueError(
                f"tokenizer padding_side must be 'left' for batched generation, got {tokenizer.padding_side!r}"
            )
        self._models = models
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._device = device
        self._pad_id = pad_id
        self._eos_id = eos_id
        self._extra = tuple(extra_logits_processors)

    def sync_weights(self, models: PolicyModels) -> None:
        """No-op: generation reads the same live module the trainer updates."""
        del models

    def generate(self, prompts: Sequence[str], *, greedy: bool) -> RolloutBatch:
        if len(prompts) == 0:
            raise ValueError("prompts must be non-empty")
        enc = self._tokenizer(list(prompts), return_tensors="pt", padding=True)
        prompt_ids: Tensor = enc.data["input_ids"].to(self._device)
        prompt_mask: Tensor = enc.data["attention_mask"].to(self._device)
        longest = int(prompt_mask.sum(dim=1).max().item())
        if longest > self._cfg.max_prompt_tokens:
            raise ValueError(f"prompt has {longest} tokens, over max_prompt_tokens={self._cfg.max_prompt_tokens}")

        chunk = self._cfg.generation_batch_size
        raw = [
            self._generate_chunk(prompt_ids[i : i + chunk], prompt_mask[i : i + chunk], greedy=greedy)
            for i in range(0, len(prompts), chunk)
        ]
        response_ids = self._right_pad(raw)
        is_eos = (response_ids == self._eos_id).long()
        # Keep tokens up to and including the first EOS (cumsum is 1 there, minus itself is 0).
        keep = (is_eos.cumsum(dim=1) - is_eos) == 0
        response_ids = torch.where(keep, response_ids, torch.full_like(response_ids, self._pad_id))
        texts = tuple(
            self._tokenizer.decode(row[m].tolist(), skip_special_tokens=True)
            for row, m in zip(response_ids, keep, strict=True)
        )
        return RolloutBatch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            response_ids=response_ids,
            response_mask=keep.long(),
            response_text=texts,
            hit_max_tokens=~(is_eos.sum(dim=1) > 0),
        )

    def _generate_chunk(self, prompt_ids: Tensor, prompt_mask: Tensor, *, greedy: bool) -> Tensor:
        module = self._models.generation_module()
        sampling = (
            {"do_sample": False, "temperature": 1.0}
            if greedy
            else {
                "do_sample": True,
                "temperature": self._cfg.temperature,
            }
        )
        with torch.no_grad():
            out = module.generate(  # pyright: ignore[reportCallIssue]
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self._cfg.max_new_tokens,
                top_p=self._cfg.top_p,
                top_k=0,
                repetition_penalty=1.0,
                pad_token_id=self._pad_id,
                eos_token_id=self._eos_id,
                logits_processor=LogitsProcessorList(self._extra),
                **sampling,
            )
        return out[:, prompt_ids.shape[1] :]

    def _right_pad(self, chunks: list[Tensor]) -> Tensor:
        width = max(c.shape[1] for c in chunks)
        padded = [torch.nn.functional.pad(c, (0, width - c.shape[1]), value=self._pad_id) for c in chunks]
        return torch.cat(padded, dim=0)
