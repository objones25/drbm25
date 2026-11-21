"""VLLMEngine: sampling parameters, prompt/response handling and weight sync.

CPU tests drive the engine with a fake ``vllm.LLM`` (vLLM is a Linux-only extra and is not
installed on the dev Mac). ``@pytest.mark.gpu`` tests build a real colocated engine and run on
the A100 pod (Task 17 smoke stage): ``uv run pytest -m gpu tests/rollout/test_vllm_engine.py``.
"""

import importlib.util
import os
import socket
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.testing import assert_close

from drbm25.config import LoraCfg, RolloutCfg
from drbm25.distributed import DistInfo
from drbm25.ppo.models import LoraPolicyModels
from drbm25.rollout.base import RolloutEngine
from drbm25.rollout.vllm_engine import VLLMEngine, VLLMUnavailableError, sampling_kwargs
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer

CPU = torch.device("cpu")
DIST = DistInfo(rank=0, world_size=1, local_rank=0, device=CPU)
TOK = build_tiny_tokenizer()
PAD, EOS = 0, 1
HAS_VLLM = importlib.util.find_spec("vllm") is not None


def _cfg(max_new_tokens: int = 4) -> RolloutCfg:
    return RolloutCfg(
        backend="vllm",
        temperature=0.6,
        top_p=1.0,
        max_prompt_tokens=8,
        max_new_tokens=max_new_tokens,
        generation_batch_size=4,
        vllm_gpu_memory_utilization=0.2,
    )


@dataclass
class _Completion:
    token_ids: list[int]
    finish_reason: str | None


@dataclass
class _RequestOutput:
    outputs: list[_Completion]


@dataclass
class FakeLLM:
    """Records calls; answers each prompt with the next scripted completion."""

    script: list[_Completion] = field(default_factory=list)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    loaded: list[tuple[str, Tensor]] = field(default_factory=list)
    param_names: set[str] | None = None  # the "model"'s parameters; None: exactly the loaded names
    config: Any = None
    prefix_cache_reset: bool = True

    def generate(
        self, prompts: Sequence[Mapping[str, list[int]]], *, sampling_params: Any, use_tqdm: bool
    ) -> list[Any]:
        assert use_tqdm is False
        self.calls.append(("generate", (list(prompts), sampling_params)))
        return [_RequestOutput([self.script.pop(0)]) for _ in prompts]

    def apply_model(self, func: Any) -> list[Any]:
        self.calls.append(("apply_model", None))
        return [func(self)]

    def load_weights(self, weights: Any) -> set[str]:  # the "model" apply_model hands to func
        self.loaded.extend(weights)
        return {name for name, _ in self.loaded}

    def named_parameters(self) -> list[tuple[str, None]]:
        names = self.param_names if self.param_names is not None else {name for name, _ in self.loaded}
        return [(name, None) for name in sorted(names)]

    def sleep(self, level: int = 1) -> None:
        self.calls.append(("sleep", level))

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.calls.append(("wake_up", tags))

    def reset_prefix_cache(self) -> bool:
        self.calls.append(("reset_prefix_cache", None))
        return self.prefix_cache_reset


def _engine(llm: FakeLLM, cfg: RolloutCfg | None = None) -> VLLMEngine:
    return VLLMEngine(llm, TOK, cfg or _cfg(), DIST, make_sampling_params=dict)


def _names(llm: FakeLLM) -> list[str]:
    return [name for name, _ in llm.calls]


# --------------------------------------------------------------------------- sampling parameters
def test_sampling_matches_the_training_distribution() -> None:
    # temperature: the trainer divides logits by rollout.temperature; every other warper is off.
    # top_k 0 disables top-k in vLLM 0.19 (-1 is also accepted).
    assert sampling_kwargs(_cfg(), greedy=False, eos_id=EOS) == {
        "n": 1,
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": 4,
        "stop_token_ids": [EOS],
        "detokenize": False,
    }


def test_greedy_uses_temperature_zero() -> None:
    kw = sampling_kwargs(_cfg(), greedy=True, eos_id=EOS)
    # vLLM: temperature < 1e-5 is greedy.
    assert (kw["temperature"], kw["top_p"]) == (0.0, 1.0)


# --------------------------------------------------------------------------- generate
def test_generate_sends_unpadded_prompt_ids_and_builds_the_batch() -> None:
    llm = FakeLLM(script=[_Completion([5, 6, EOS], "stop"), _Completion([7, 8, 9, 10], "length")])
    batch = _engine(llm).generate(["w0 w1 w2", "w3"], greedy=False)
    (prompts, params) = llm.calls[0][1]
    assert prompts == [{"prompt_token_ids": [3, 4, 5]}, {"prompt_token_ids": [6]}]
    assert params == sampling_kwargs(_cfg(), greedy=False, eos_id=EOS)
    assert batch.prompt_ids.tolist() == [[3, 4, 5], [PAD, PAD, 6]]
    assert batch.prompt_mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert batch.response_ids.tolist() == [[5, 6, EOS, PAD], [7, 8, 9, 10]]
    assert batch.response_mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert batch.hit_max_tokens.tolist() == [False, True]
    assert batch.response_text == ("w2 w3", "w4 w5 w6 w7")


def test_generate_greedy_passes_greedy_params() -> None:
    llm = FakeLLM(script=[_Completion([EOS], "stop")])
    _engine(llm).generate(["w0"], greedy=True)
    assert llm.calls[0][1][1] == sampling_kwargs(_cfg(), greedy=True, eos_id=EOS)


@pytest.mark.parametrize(
    ("completion", "match"),
    [
        (_Completion([5, 6], "stop"), "did not end with EOS"),
        (_Completion([5, 6], "length"), "2 tokens, expected max_tokens=4"),
        (_Completion([5], None), "unexpected finish_reason None"),
        (_Completion([5], "abort"), "unexpected finish_reason 'abort'"),
    ],
)
def test_generate_rejects_inconsistent_vllm_output(completion: _Completion, match: str) -> None:
    llm = FakeLLM(script=[completion])
    with pytest.raises(RuntimeError, match=match):
        _engine(llm).generate(["w0"], greedy=False)


def test_generate_rejects_overlong_prompt() -> None:
    with pytest.raises(ValueError, match="over max_prompt_tokens"):
        _engine(FakeLLM()).generate([" ".join(["w0"] * 9)], greedy=False)


# --------------------------------------------------------------------------- sleep mode and weight sync
def test_engine_sleeps_after_generate_and_wakes_weights_and_kv_cache_before_the_next() -> None:
    llm = FakeLLM(script=[_Completion([EOS], "stop"), _Completion([EOS], "stop")])
    engine = _engine(llm)
    engine.generate(["w0"], greedy=False)
    engine.generate(["w0"], greedy=False)
    assert llm.calls[0][0] == "generate"  # awake at start: weights are the checkpoint's
    assert _names(llm)[1:] == ["sleep", "wake_up", "wake_up", "generate", "sleep"]
    assert llm.calls[1][1] == 1  # level 1: weights offloaded to CPU, KV cache dropped
    # No sync in between: the offloaded weights must come back too, not just the KV cache.
    assert [llm.calls[2][1], llm.calls[3][1]] == [["weights"], ["kv_cache"]]


def test_sync_weights_loads_merged_actor_into_a_sleeping_engine() -> None:
    llm = FakeLLM(script=[_Completion([EOS], "stop"), _Completion([EOS], "stop")])
    engine = _engine(llm)
    engine.generate(["w0"], greedy=False)
    models = LoraPolicyModels(build_tiny_qwen2(0), LoraCfg(4, 8, 0.0, ("q_proj",)), gradient_checkpointing=False)
    engine.sync_weights(models)
    engine.generate(["w0"], greedy=False)
    assert _names(llm) == [
        "generate",
        "sleep",
        "wake_up",  # weights only: the KV cache stays released until generate
        "apply_model",
        "reset_prefix_cache",
        "wake_up",
        "generate",
        "sleep",
    ]
    assert llm.calls[2][1] == ["weights"]
    assert llm.calls[5][1] == ["kv_cache"]
    want = models.merged_actor_state_dict()
    assert [name for name, _ in llm.loaded] == list(want)
    assert_close(dict(llm.loaded), want, rtol=0, atol=0)


def test_sync_weights_on_an_awake_engine_skips_wake_up() -> None:
    llm = FakeLLM()
    _engine(llm).sync_weights(
        LoraPolicyModels(build_tiny_qwen2(0), LoraCfg(4, 8, 0.0, ("q_proj",)), gradient_checkpointing=False)
    )
    assert _names(llm) == ["apply_model", "reset_prefix_cache"]


@dataclass
class _Weights:
    weights: dict[str, Tensor]

    def merged_actor_state_dict(self) -> dict[str, Tensor]:
        return self.weights


@dataclass
class _TiedConfig:
    tie_word_embeddings: bool


def test_sync_weights_raises_on_a_partial_sync() -> None:
    llm = FakeLLM(param_names={"a.weight", "b.weight"})
    with pytest.raises(RuntimeError, match=r"partial weight sync.*'b\.weight'"):
        _engine(llm).sync_weights(_Weights({"a.weight": torch.zeros(1)}))
    assert "reset_prefix_cache" not in _names(llm)


def test_sync_weights_allows_a_tied_lm_head_to_be_skipped() -> None:
    llm = FakeLLM(param_names={"a.weight", "lm_head.weight"}, config=_TiedConfig(tie_word_embeddings=True))
    _engine(llm).sync_weights(_Weights({"a.weight": torch.zeros(1)}))
    assert _names(llm) == ["apply_model", "reset_prefix_cache"]


def test_sync_weights_requires_an_untied_lm_head() -> None:
    llm = FakeLLM(param_names={"a.weight", "lm_head.weight"}, config=_TiedConfig(tie_word_embeddings=False))
    with pytest.raises(RuntimeError, match=r"lm_head\.weight"):
        _engine(llm).sync_weights(_Weights({"a.weight": torch.zeros(1)}))


def test_sync_weights_raises_when_the_prefix_cache_is_not_reset() -> None:
    llm = FakeLLM(prefix_cache_reset=False)
    with pytest.raises(RuntimeError, match="prefix cache"):
        _engine(llm).sync_weights(_Weights({"a.weight": torch.zeros(1)}))


def test_engine_satisfies_protocol() -> None:
    engine = _engine(FakeLLM(script=[_Completion([EOS], "stop")]))
    assert isinstance(engine, RolloutEngine)
    assert engine.generate(["w5"], greedy=True).response_ids.tolist() == [[EOS]]


@pytest.mark.skipif(HAS_VLLM, reason="vllm is installed here")
def test_create_without_vllm_says_how_to_install() -> None:
    with pytest.raises(VLLMUnavailableError, match="--extra vllm"):
        VLLMEngine.create("m", "main", _cfg(), DIST, tokenizer=TOK, dtype="bfloat16", seed=0)


# --------------------------------------------------------------------------- GPU (A100 pod)
GPU_MODEL = os.environ.get("DRBM25_VLLM_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
GPU_PROMPTS = (
    "What is the boiling point of water at sea level?",
    "List three uses of copper.",
    "Rewrite this search query for a medical database: heart attack aspirin",
)


@pytest.fixture(scope="module")
def gpu_engine() -> Iterator[tuple[VLLMEngine, Any]]:
    from transformers import AutoTokenizer

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.MonkeyPatch.context() as mp:
        # external_launcher reads the torchrun variables; a one-rank "launch" for the test process.
        for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1"}.items():
            mp.setenv(key, value)
        mp.setenv("MASTER_PORT", str(port))
        tokenizer = AutoTokenizer.from_pretrained(GPU_MODEL, padding_side="left")
        cfg = RolloutCfg("vllm", 0.6, 1.0, 256, 20, 32, 0.3)
        dist = DistInfo(0, 1, 0, torch.device("cuda", 0))  # lint: ignore[CUDA_HARDCODE] one-GPU test
        engine = VLLMEngine.create(GPU_MODEL, "main", cfg, dist, tokenizer=tokenizer, dtype="bfloat16", seed=0)
        yield engine, tokenizer


@dataclass
class _StaticActor:
    """Only what ``VLLMEngine.sync_weights`` may use (controller ruling R4)."""

    weights: dict[str, Tensor]

    def merged_actor_state_dict(self) -> dict[str, Tensor]:
        return self.weights


@pytest.mark.gpu
def test_vllm_generate_matches_hf_greedy_on_3_prompts(gpu_engine: tuple[VLLMEngine, Any]) -> None:
    from transformers import AutoModelForCausalLM

    from drbm25.prompt import build_prompt

    engine, tokenizer = gpu_engine
    prompts = [build_prompt(q) for q in GPU_PROMPTS]
    got = engine.generate(prompts, greedy=True)
    hf = AutoModelForCausalLM.from_pretrained(GPU_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    nn.Module.to(hf, torch.device("cuda"))  # lint: ignore[CUDA_HARDCODE] one-GPU test
    for i, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = hf.generate(**enc, max_new_tokens=20, do_sample=False, top_k=0, top_p=1.0, repetition_penalty=1.0)
        want = out[0, enc["input_ids"].shape[1] :].tolist()
        n = min(20, len(want))
        assert got.response_ids[i, :n].tolist() == want[:n], f"prompt {i}"
    del hf
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_vllm_sync_weights_changes_output_after_update(gpu_engine: tuple[VLLMEngine, Any]) -> None:
    from transformers import AutoModelForCausalLM

    from drbm25.prompt import build_prompt

    engine, _ = gpu_engine
    prompts = [build_prompt(q) for q in GPU_PROMPTS]
    before = engine.generate(prompts, greedy=True).response_ids
    hf = AutoModelForCausalLM.from_pretrained(GPU_MODEL, dtype=torch.bfloat16)
    nn.Module.to(hf, torch.device("cuda"))  # lint: ignore[CUDA_HARDCODE] one-GPU test
    original = {k: v.detach().clone() for k, v in hf.state_dict().items()}
    gen = torch.Generator(device="cuda").manual_seed(0)
    perturbed = {
        k: v + 0.05 * torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype) for k, v in original.items()
    }
    del hf
    engine.sync_weights(_StaticActor(perturbed))  # R4: only merged_actor_state_dict is used
    changed = engine.generate(prompts, greedy=True).response_ids
    assert not torch.equal(changed, before)
    engine.sync_weights(_StaticActor(original))
    restored = engine.generate(prompts, greedy=True).response_ids
    assert torch.equal(restored, before)
