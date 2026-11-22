import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import drbm25.eval.sample as sample_mod
from drbm25.config import load_config
from drbm25.data import DEV_FILE
from drbm25.eval.sample import generate_samples, main
from drbm25.eval.sample_db import read_db, shard_complete
from drbm25.prompt import build_prompt
from drbm25.tiny import build_tiny_tokenizer

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"
TOK = build_tiny_tokenizer()
EOS = TOK.eos_token_id


@dataclass
class _Logprob:
    logprob: float


@dataclass
class _Completion:
    token_ids: list[int]
    finish_reason: str
    logprobs: list[dict[int, _Logprob]] | None


@dataclass
class _RequestOutput:
    outputs: list[_Completion]


@dataclass
class FakeLLM:
    """Answers every request with ``n`` completions of ``w5 w6 <eos>``; records prompts and params."""

    calls: list[tuple[list[Mapping[str, list[int]]], Any]] = field(default_factory=list)
    with_logprobs: bool = True

    def generate(
        self, prompts: Sequence[Mapping[str, list[int]]], *, sampling_params: Any, use_tqdm: bool
    ) -> list[Any]:
        assert use_tqdm is False
        self.calls.append((list(prompts), sampling_params))
        out = []
        for params in sampling_params:
            tokens = [8, 9, EOS]
            lps = [{t: _Logprob(-0.5)} for t in tokens] if self.with_logprobs else None
            out.append(_RequestOutput([_Completion(tokens, "stop", lps) for _ in range(params["n"])]))
        return out


def test_generate_samples_requests_n_samples_with_training_sampling_params() -> None:
    llm = FakeLLM()
    cfg = load_config(SMOKE).rollout

    generate_samples(llm, TOK, [build_prompt("w5")], [77], cfg, n=3, make_sampling_params=dict)

    ((_, params),) = llm.calls
    (p,) = params
    assert p["n"] == 3
    assert p["temperature"] == pytest.approx(0.6)
    assert p["top_p"] == pytest.approx(1.0)
    assert p["top_k"] == 0
    assert p["logprobs"] == 0
    assert p["seed"] == 77
    assert p["stop_token_ids"] == [EOS]
    assert p["max_tokens"] == cfg.max_new_tokens


def test_generate_samples_sends_training_prompt_tokens() -> None:
    llm = FakeLLM()

    generate_samples(llm, TOK, [build_prompt("w5")], [1], load_config(SMOKE).rollout, n=1, make_sampling_params=dict)

    ((prompts, _),) = llm.calls
    assert prompts[0]["prompt_token_ids"] == TOK(build_prompt("w5")).input_ids


def test_generate_samples_decodes_with_training_tokenizer() -> None:
    out = generate_samples(
        FakeLLM(), TOK, [build_prompt("w5")], [1], load_config(SMOKE).rollout, n=2, make_sampling_params=dict
    )

    assert [[g.text for g in samples] for samples in out] == [["w5 w6", "w5 w6"]]
    assert out[0][0].n_tokens == 3
    assert out[0][0].hit_max_tokens is False


def test_sum_logprob_is_sum_of_sampled_token_logprobs() -> None:
    (samples,) = generate_samples(
        FakeLLM(), TOK, [build_prompt("w5")], [1], load_config(SMOKE).rollout, n=1, make_sampling_params=dict
    )

    assert samples[0].sum_logprob == pytest.approx(-1.5)


def test_sum_logprob_none_when_engine_returns_no_logprobs() -> None:
    (samples,) = generate_samples(
        FakeLLM(with_logprobs=False),
        TOK,
        [build_prompt("w5")],
        [1],
        load_config(SMOKE).rollout,
        n=1,
        make_sampling_params=dict,
    )

    assert samples[0].sum_logprob is None


# ---------------------------------------------------------------- CLI (needs the toy index and Java)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "data" / DEV_FILE
    path.parent.mkdir(parents=True)
    rows = [{"question": f"solar efficiency {i}", "docs_id": ["docB"]} for i in range(5)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return tmp_path


def _argv(workdir: Path, toy_index: Path, out: Path, *extra: str) -> list[str]:
    sets = {"retrieval.index_dir": str(toy_index), "data.data_dir": str(workdir / "data")}
    argv = ["--config", str(SMOKE), "--out-dir", str(out)]
    for key, value in sets.items():
        argv += ["--set", f"{key}={value}"]
    return argv + list(extra)


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(sample_mod, "_build_llm", lambda *_args: (llm, dict))
    monkeypatch.setattr(sample_mod, "_resolve_revision", lambda *_args: "resolved-sha")
    return llm


@pytest.mark.java
def test_cli_original_source_needs_no_model(workdir: Path, toy_index: Path, fake_llm: FakeLLM) -> None:
    out = workdir / "db"

    assert main(_argv(workdir, toy_index, out, "--source", "original", "--n", "4")) == 0

    rows = read_db(out)
    assert [(r.source, r.qid, r.sample_index) for r in rows] == [("original", f"dev-{i}", 0) for i in range(1, 6)]
    assert fake_llm.calls == []


@pytest.mark.java
def test_cli_model_source_writes_n_rows_per_query_with_provenance(
    workdir: Path, toy_index: Path, fake_llm: FakeLLM
) -> None:
    out = workdir / "db"

    argv = _argv(workdir, toy_index, out, "--source", "trained-normal", "--n", "2", "--shard-size", "2", "--limit", "3")
    assert main(argv) == 0

    rows = read_db(out)
    assert len(rows) == 6
    assert {r.model_id for r in rows} == {"objones25/drbm25-qwen2.5-3b-normal"}
    assert {r.model_revision for r in rows} == {"resolved-sha"}
    assert [r.seed for r in rows if r.sample_index == 0] == [1234 * 1_000_003 + i for i in range(3)]
    assert sorted(p.name for p in (out / "trained-normal").glob("shard-*.parquet")) == [
        "shard-00000.parquet",
        "shard-00001.parquet",
    ]


@pytest.mark.java
def test_cli_resumes_skipping_complete_shards(workdir: Path, toy_index: Path, fake_llm: FakeLLM) -> None:
    out = workdir / "db"
    argv = _argv(workdir, toy_index, out, "--source", "untrained", "--n", "1", "--shard-size", "2")
    assert main(argv) == 0
    shard1 = out / "untrained" / "shard-00001.parquet"
    shard1.with_name(shard1.name + "._SUCCESS").unlink()  # as if the run died while writing shard 1
    fake_llm.calls.clear()

    assert main(argv) == 0

    seen = [len(prompts) for prompts, _ in fake_llm.calls]
    assert seen == [2]  # only shard 1 (queries 2-3) was generated again
    assert shard_complete(shard1)
    assert len(read_db(out)) == 5


def test_cli_rejects_unknown_source(workdir: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--config", str(SMOKE), "--out-dir", str(workdir / "db"), "--source", "bogus", "--n", "1"])
