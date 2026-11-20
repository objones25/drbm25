"""HFGenerateEngine: padding layout, EOS-aware response masks, determinism, input checks."""

import pytest
import torch
from torch import Tensor
from transformers import LogitsProcessor, PreTrainedTokenizerFast

from drbm25.config import LoraCfg, RolloutCfg
from drbm25.ppo.models import LoraPolicyModels
from drbm25.rollout.base import RolloutBatch, RolloutEngine
from drbm25.rollout.hf_generate import HFGenerateEngine
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer

LORA = LoraCfg(rank=4, alpha=8, dropout=0.0, target_modules=("q_proj", "v_proj"))
DEVICE = torch.device("cpu")


def _cfg(*, max_prompt_tokens: int = 8, max_new_tokens: int = 6, generation_batch_size: int = 4) -> RolloutCfg:
    return RolloutCfg(
        backend="hf",
        temperature=0.6,
        top_p=1.0,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        generation_batch_size=generation_batch_size,
        vllm_gpu_memory_utilization=0.0,
    )


def _models() -> LoraPolicyModels:
    return LoraPolicyModels(build_tiny_qwen2(0), LORA, gradient_checkpointing=False)


def _engine(
    tokenizer: PreTrainedTokenizerFast | None = None,
    cfg: RolloutCfg | None = None,
    extra: tuple[LogitsProcessor, ...] = (),
) -> HFGenerateEngine:
    return HFGenerateEngine(
        _models(),
        tokenizer or build_tiny_tokenizer(),
        cfg or _cfg(),
        DEVICE,
        extra_logits_processors=extra,
    )


class _ForceEosRow0(LogitsProcessor):
    """Row 0 emits EOS as its 3rd new token; every other row can never emit EOS."""

    def __init__(self, prompt_len: int, eos_id: int, eos_step: int = 3) -> None:
        self._prompt_len = prompt_len
        self._eos_id = eos_id
        self._eos_step = eos_step

    def __call__(self, input_ids: Tensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self._prompt_len + 1
        scores[1:, self._eos_id] = float("-inf")
        if step == self._eos_step:
            scores[0] = float("-inf")
            scores[0, self._eos_id] = 0.0
        elif step < self._eos_step:
            scores[0, self._eos_id] = float("-inf")
        return scores


class _NeverEos(LogitsProcessor):
    def __init__(self, eos_id: int) -> None:
        self._eos_id = eos_id

    def __call__(self, input_ids: Tensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        scores[:, self._eos_id] = float("-inf")
        return scores


def _special_ids() -> tuple[int, int]:
    tok = build_tiny_tokenizer()
    pad, eos = tok.pad_token_id, tok.eos_token_id
    assert isinstance(pad, int)
    assert isinstance(eos, int)
    return pad, eos


def test_tiny_tokenizer_matches_tiny_model() -> None:
    tok = build_tiny_tokenizer()
    assert len(tok) == 128
    pad, eos = _special_ids()
    assert pad != eos
    assert tok.padding_side == "left"


def test_shapes_and_left_padding() -> None:
    engine = _engine()
    batch = engine.generate(["w5 w6 w7", "w9", "w10 w11"], greedy=True)
    assert isinstance(batch, RolloutBatch)
    assert tuple(batch.prompt_ids.shape) == (3, 3)
    assert batch.prompt_mask.tolist() == [[1, 1, 1], [0, 0, 1], [0, 1, 1]]
    pad, _ = _special_ids()
    assert batch.prompt_ids[1, :2].tolist() == [pad, pad]
    assert batch.response_ids.shape == batch.response_mask.shape
    assert batch.response_ids.shape[0] == 3
    assert 1 <= batch.response_ids.shape[1] <= 6
    assert len(batch.response_text) == 3
    assert tuple(batch.hit_max_tokens.shape) == (3,)
    assert batch.hit_max_tokens.dtype == torch.bool


def test_chunks_are_merged_in_order() -> None:
    prompts = ["w5", "w6 w7", "w8", "w9 w10 w11", "w12"]
    small = _engine(cfg=_cfg(generation_batch_size=2)).generate(prompts, greedy=True)
    large = _engine(cfg=_cfg(generation_batch_size=8)).generate(prompts, greedy=True)
    assert small.prompt_ids.tolist() == large.prompt_ids.tolist()
    assert small.response_text == large.response_text
    assert small.response_ids.shape[0] == 5


def test_response_mask_stops_after_eos() -> None:
    pad, eos = _special_ids()
    # Two prompts of 2 tokens each: no left padding, so prompt_len is 2.
    engine = _engine(extra=(_ForceEosRow0(prompt_len=2, eos_id=eos),))
    batch = engine.generate(["w5 w6", "w7 w8"], greedy=True)
    assert batch.response_mask[0].sum().item() == 3
    assert batch.response_ids[0, 2].item() == eos
    assert batch.response_mask[0, :3].tolist() == [1, 1, 1]
    assert batch.response_mask[0, 3:].tolist() == [0] * (batch.response_ids.shape[1] - 3)
    assert batch.response_ids[0, 3:].tolist() == [pad] * (batch.response_ids.shape[1] - 3)
    assert batch.hit_max_tokens.tolist() == [False, True]
    assert batch.response_mask[1].sum().item() == 6


def test_hit_max_tokens_flag() -> None:
    _, eos = _special_ids()
    engine = _engine(extra=(_NeverEos(eos),))
    batch = engine.generate(["w5 w6", "w7"], greedy=False)
    assert batch.hit_max_tokens.tolist() == [True, True]
    assert batch.response_mask.sum(dim=1).tolist() == [6, 6]


def test_greedy_is_deterministic() -> None:
    engine = _engine()
    a = engine.generate(["w5 w6", "w7"], greedy=True)
    b = engine.generate(["w5 w6", "w7"], greedy=True)
    assert torch.equal(a.response_ids, b.response_ids)
    assert a.response_text == b.response_text


def test_sampling_is_reproducible_given_seed() -> None:
    engine = _engine()
    torch.manual_seed(123)
    a = engine.generate(["w5 w6", "w7"], greedy=False)
    torch.manual_seed(123)
    b = engine.generate(["w5 w6", "w7"], greedy=False)
    assert torch.equal(a.response_ids, b.response_ids)


def test_response_text_skips_special_tokens() -> None:
    _, eos = _special_ids()
    engine = _engine(extra=(_ForceEosRow0(prompt_len=2, eos_id=eos),))
    batch = engine.generate(["w5 w6", "w7 w8"], greedy=True)
    assert "<eos>" not in batch.response_text[0]
    assert "<pad>" not in batch.response_text[0]
    assert len(batch.response_text[0].split()) == 2


def test_generation_leaves_models_usable_for_actor_forward() -> None:
    models = _models()
    engine = HFGenerateEngine(models, build_tiny_tokenizer(), _cfg(), DEVICE)
    engine.generate(["w5 w6"], greedy=True)
    ids = torch.randint(0, 128, (1, 4))
    assert models.actor_logits(ids, torch.ones_like(ids)).requires_grad


def test_outputs_do_not_require_grad_and_are_not_inference_tensors() -> None:
    batch = _engine().generate(["w5"], greedy=True)
    assert not batch.response_ids.is_inference()
    (batch.response_ids.float() * torch.ones(1, requires_grad=True)).sum().backward()


def test_sync_weights_is_noop() -> None:
    models = _models()
    engine = HFGenerateEngine(models, build_tiny_tokenizer(), _cfg(), DEVICE)
    assert engine.sync_weights(models) is None


def test_engine_satisfies_protocol() -> None:
    engine = _engine()
    assert isinstance(engine, RolloutEngine)
    assert engine.generate(["w5"], greedy=True).response_ids.shape[0] == 1


def test_rejects_overlong_prompt() -> None:
    engine = _engine(cfg=_cfg(max_prompt_tokens=3))
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        engine.generate(["w5 w6 w7 w8"], greedy=True)


def test_rejects_empty_prompt_list() -> None:
    with pytest.raises(ValueError, match="prompts"):
        _engine().generate([], greedy=True)


def test_rejects_pad_equal_eos_tokenizer() -> None:
    tok = build_tiny_tokenizer()
    tok.pad_token = tok.eos_token
    with pytest.raises(ValueError, match="pad_token_id"):
        _engine(tokenizer=tok)


def test_rejects_right_padding_tokenizer() -> None:
    tok = build_tiny_tokenizer()
    tok.padding_side = "right"
    with pytest.raises(ValueError, match="padding_side"):
        _engine(tokenizer=tok)
