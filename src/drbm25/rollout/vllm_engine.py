"""Colocated vLLM rollout engine for the A100 layout: one ``vllm.LLM`` per rank, TP = 1.

Each rank generates its own share of the prompts (data parallel). After every PPO update
``sync_weights`` pushes the full actor (``models.merged_actor_state_dict()``, the only call it
makes on the models: controller ruling R4) into the engine's model in-process. Between
rollouts the engine sleeps (level 1: weights offloaded to host memory, KV cache released), so
training gets the GPU memory the KV cache held.

vLLM is a Linux-only extra and is imported lazily, in :meth:`VLLMEngine.create`. API facts read
from the vLLM v0.19.0 source (``raw.githubusercontent.com/vllm-project/vllm/v0.19.0/...``):
  * ``LLM(model, *, tensor_parallel_size, dtype, revision, tokenizer_revision, seed,
    gpu_memory_utilization, enforce_eager, **EngineArgs)`` (``entrypoints/llm.py``);
    ``EngineArgs`` has ``max_model_len``, ``enable_sleep_mode``, ``generation_config``,
    ``distributed_executor_backend`` (``engine/arg_utils.py``).
  * ``distributed_executor_backend="external_launcher"`` runs the engine core and its single
    worker in this process (``config/parallel.py`` sets ``VLLM_ENABLE_V1_MULTIPROCESSING=0``;
    ``ExecutorWithExternalLauncher`` reads ``RANK`` / ``LOCAL_RANK`` from torchrun), reuses the
    existing torch process group, and with TP = 1 treats the ranks as independent "external
    DP" engines (``distributed/parallel_state.py``). Construction is a collective: KV-cache
    memory is the min over ranks.
  * ``LLM.apply_model(fn)`` -> ``collective_rpc("apply_model")`` -> ``Worker.apply_model`` runs
    ``fn(model_runner.get_model())`` in-process (``v1/executor/uniproc_executor.py``,
    ``v1/worker/worker_base.py``), so tensors are passed by reference.
    ``Qwen2ForCausalLM.load_weights(Iterable[(hf_name, tensor)])`` maps HF names (fusing
    q/k/v and gate/up), skips ``lm_head.`` when embeddings are tied, and returns the set of vLLM
    parameter names it wrote (``model_executor/models/qwen2.py``, ``AutoWeightsLoader`` in
    ``models/utils.py``). TRL's colocate mode (v1.10.0, which supports
    vllm>=0.17,<=0.26) loads weights through the same model object.
  * ``LLM.sleep(level=1)`` offloads weights to CPU and discards the KV cache;
    ``wake_up(tags)`` takes ``"weights"`` / ``"kv_cache"``; ``reset_prefix_cache()``. Sleep mode
    uses the CuMem allocator, which refuses ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.
  * ``SamplingParams``: ``top_k=0`` (or -1) disables top-k; ``temperature < 1e-5`` is greedy;
    the defaults from the model's ``generation_config.json`` apply only when no
    ``SamplingParams`` is passed, and ``generation_config="vllm"`` ignores that file anyway.
  * A request that stops on EOS (or a ``stop_token_ids`` entry) has that token as the last
    of ``CompletionOutput.token_ids`` (``v1/core/sched/utils.py: check_stop``).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, Protocol

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from drbm25.config import RolloutCfg
from drbm25.distributed import DistInfo
from drbm25.rollout.base import RolloutBatch, encode_prompts, finalize_rollout, right_pad, special_ids

logger = logging.getLogger(__name__)

VLLMDtype = Literal["float16", "bfloat16", "float32"]
_SLEEP_LEVEL: Final = 1  # weights to host memory; level 2 would discard them
_WEIGHTS: Final = "weights"
_KV_CACHE: Final = "kv_cache"


class VLLMUnavailableError(RuntimeError):
    """vLLM is not importable in this environment (it is a Linux-only optional extra)."""


class _LLM(Protocol):
    """The part of ``vllm.LLM`` (0.19) this engine uses."""

    def generate(
        self, prompts: Sequence[Mapping[str, list[int]]], *, sampling_params: Any, use_tqdm: bool
    ) -> list[Any]: ...

    def apply_model(self, func: Callable[[Any], Any]) -> list[Any]: ...

    def sleep(self, level: int = 1) -> None: ...

    def wake_up(self, tags: list[str] | None = None) -> None: ...

    def reset_prefix_cache(self) -> bool: ...


class MergedActor(Protocol):
    """All ``sync_weights`` may read from the models (R4)."""

    def merged_actor_state_dict(self) -> dict[str, Tensor]: ...


def sampling_kwargs(cfg: RolloutCfg, *, greedy: bool, eos_id: int) -> dict[str, Any]:
    """``SamplingParams`` fields that make vLLM sample from ``softmax(logits / T)`` and nothing else.

    The trainer scores old/ref/new log-probs from ``logits / rollout.temperature`` with the full
    vocabulary, so every other warper is disabled: top-k off, min-p 0, no penalties.
    """
    return {
        "n": 1,
        "temperature": 0.0 if greedy else cfg.temperature,
        "top_p": 1.0 if greedy else cfg.top_p,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": cfg.max_new_tokens,
        "stop_token_ids": [eos_id],
        "detokenize": False,  # response text is decoded with the training tokenizer
    }


class VLLMEngine:
    """``RolloutEngine`` over a colocated ``vllm.LLM``; build it with :meth:`create`.

    The constructor takes the engine object (tests pass a fake). Prompts are tokenized with the
    training tokenizer and sent as token ids, so vLLM sees exactly the trainer's prompt tokens;
    the response ids come back from vLLM unchanged. Not thread-safe.
    """

    def __init__(
        self,
        llm: _LLM,
        tokenizer: PreTrainedTokenizerBase,
        cfg: RolloutCfg,
        dist: DistInfo,
        *,
        make_sampling_params: Callable[..., Any],
    ) -> None:
        self._llm = llm
        self._tokenizer = tokenizer
        self._ids = special_ids(tokenizer)
        self._cfg = cfg
        self._dist = dist
        self._make_sampling_params = make_sampling_params
        self._asleep: set[str] = set()

    @classmethod
    def create(
        cls,
        model_name: str,
        revision: str,
        cfg: RolloutCfg,
        dist: DistInfo,
        *,
        tokenizer: PreTrainedTokenizerBase,
        dtype: VLLMDtype,
        seed: int,
    ) -> VLLMEngine:
        """Build the per-rank engine. A collective under torchrun: call on every rank, before the
        training models take GPU memory (vLLM checks at start-up that
        ``gpu_memory_utilization`` x total memory is free)."""
        try:
            vllm = importlib.import_module("vllm")
        except ImportError as exc:
            raise VLLMUnavailableError(
                "rollout.backend 'vllm' needs vLLM 0.19 (Linux only): install it with 'uv sync --extra vllm'"
            ) from exc
        llm = vllm.LLM(
            model=model_name,
            revision=revision,
            tokenizer_revision=revision,
            tensor_parallel_size=1,
            dtype=dtype,
            seed=seed,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            max_model_len=cfg.max_prompt_tokens + cfg.max_new_tokens,
            enforce_eager=True,  # no CUDA graphs: saves their memory, as upstream verl did
            enable_sleep_mode=True,
            generation_config="vllm",
            distributed_executor_backend="external_launcher",
        )
        logger.info("vLLM engine ready on rank %d (%s, dtype %s)", dist.rank, model_name, dtype)
        return cls(llm, tokenizer, cfg, dist, make_sampling_params=vllm.SamplingParams)

    # ------------------------------------------------------------------ sleep mode
    def _wake(self, tag: str) -> None:
        if tag in self._asleep:
            if self._dist.device.type == "cuda":
                torch.cuda.empty_cache()  # return the trainer's cached blocks so vLLM can map its memory
            self._llm.wake_up(tags=[tag])
            self._asleep.discard(tag)

    def _sleep(self) -> None:
        self._llm.sleep(level=_SLEEP_LEVEL)
        self._asleep = {_WEIGHTS, _KV_CACHE}

    # ------------------------------------------------------------------ RolloutEngine
    def sync_weights(self, models: MergedActor) -> None:
        """Load ``models.merged_actor_state_dict()`` into the engine's model (a collective on FSDP).

        Raises ``RuntimeError`` if any vLLM parameter was not written (a tied ``lm_head`` is the
        only allowed gap) or if the prefix cache could not be reset: either would leave rollouts
        sampling from a mix of old and new weights without any visible error.
        """
        weights = models.merged_actor_state_dict()
        self._wake(_WEIGHTS)

        def load(model: Any) -> list[str]:
            loaded = set(model.load_weights(weights.items()))  # vLLM parameter names it wrote
            expected = {name for name, _ in model.named_parameters()}
            if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
                expected = {name for name in expected if not name.startswith("lm_head.")}
            return sorted(expected - loaded)

        (missing,) = self._llm.apply_model(load)
        if missing:
            raise RuntimeError(f"partial weight sync: {len(missing)} vLLM parameters not loaded, e.g. {missing[:10]}")
        if not self._llm.reset_prefix_cache():  # cached KV blocks were computed with the old weights
            raise RuntimeError("vLLM could not reset its prefix cache after the weight sync")

    def generate(self, prompts: Sequence[str], *, greedy: bool) -> RolloutBatch:
        prompt_ids, prompt_mask = encode_prompts(
            self._tokenizer, prompts, self._dist.device, max_prompt_tokens=self._cfg.max_prompt_tokens
        )
        requests = [
            {"prompt_token_ids": row[m.bool()].tolist()} for row, m in zip(prompt_ids, prompt_mask, strict=True)
        ]
        params = self._make_sampling_params(**sampling_kwargs(self._cfg, greedy=greedy, eos_id=self._ids.eos))
        self._wake(_WEIGHTS)
        self._wake(_KV_CACHE)
        outputs = self._llm.generate(requests, sampling_params=params, use_tqdm=False)
        self._sleep()
        if len(outputs) != len(requests):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(requests)} prompts")
        rows = [torch.tensor(self._response_tokens(out), dtype=torch.long) for out in outputs]
        response_ids = right_pad(rows, self._ids.pad).to(self._dist.device)
        return finalize_rollout(prompt_ids, prompt_mask, response_ids, tokenizer=self._tokenizer, ids=self._ids)

    def _response_tokens(self, request_output: Any) -> list[int]:
        """Token ids of the single completion, checked against how it finished."""
        completion = request_output.outputs[0]
        tokens = list(completion.token_ids)
        reason = completion.finish_reason
        if reason == "stop":
            if not tokens or tokens[-1] != self._ids.eos:
                raise RuntimeError(f"vLLM completion stopped but did not end with EOS {self._ids.eos}: {tokens[-5:]}")
        elif reason == "length":
            if len(tokens) != self._cfg.max_new_tokens:
                raise RuntimeError(
                    f"vLLM completion hit its length limit with {len(tokens)} tokens, expected "
                    f"max_tokens={self._cfg.max_new_tokens} (max_model_len too small?)"
                )
        else:
            raise RuntimeError(f"vLLM completion has unexpected finish_reason {reason!r}")
        return tokens
