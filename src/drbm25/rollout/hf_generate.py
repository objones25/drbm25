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
  * ``generate(synced_gpus=True)`` keeps calling forward on a rank whose rows are all
    finished until every rank is done (``_has_unfinished_sequences`` all-reduces a flag),
    so FSDP's per-forward all-gathers stay paired; checked on a 2-rank gloo FSDP2 actor.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase

from drbm25.config import RolloutCfg
from drbm25.ppo.models import PolicyModels
from drbm25.rollout.base import RolloutBatch, encode_prompts, finalize_rollout, right_pad, special_ids


class HFGenerateEngine:
    """Samples from the live actor module; ``sync_weights`` is a no-op (same module).

    Each ``generate`` call fetches ``models.generation_module()`` afresh (it activates the
    actor in eval mode and refuses while a checkpointed backward is pending), so there is
    nothing to restore afterwards. It runs under ``torch.no_grad`` rather than
    ``inference_mode`` so the returned tensors can feed grad-enabled forwards later.
    ``synced_gpus=True`` is required when the actor is FSDP-sharded: every forward is then a
    collective, and transformers keeps each rank stepping until all ranks have finished.
    ``extra_logits_processors`` exists for tests (forcing or banning tokens).
    """

    def __init__(
        self,
        models: PolicyModels,
        tokenizer: PreTrainedTokenizerBase,
        cfg: RolloutCfg,
        device: torch.device,
        *,
        synced_gpus: bool = False,
        extra_logits_processors: Sequence[LogitsProcessor] = (),
    ) -> None:
        self._ids = special_ids(tokenizer)
        self._models = models
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._device = device
        self._synced_gpus = synced_gpus
        self._extra = tuple(extra_logits_processors)

    def sync_weights(self, models: PolicyModels) -> None:
        """No-op: generation reads the same live module the trainer updates."""
        del models

    def generate(self, prompts: Sequence[str], *, greedy: bool) -> RolloutBatch:
        prompt_ids, prompt_mask = encode_prompts(
            self._tokenizer, prompts, self._device, max_prompt_tokens=self._cfg.max_prompt_tokens
        )
        chunk = self._cfg.generation_batch_size
        raw = [
            self._generate_chunk(prompt_ids[i : i + chunk], prompt_mask[i : i + chunk], greedy=greedy)
            for i in range(0, len(prompts), chunk)
        ]
        response_ids = right_pad(raw, self._ids.pad)
        return finalize_rollout(prompt_ids, prompt_mask, response_ids, tokenizer=self._tokenizer, ids=self._ids)

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
                pad_token_id=self._ids.pad,
                eos_token_id=self._ids.eos,
                logits_processor=LogitsProcessorList(self._extra),
                synced_gpus=self._synced_gpus,
                **sampling,
            )
        return out[:, prompt_ids.shape[1] :]
