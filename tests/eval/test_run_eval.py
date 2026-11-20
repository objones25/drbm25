"""Scoring original queries and rewrites, and the run_eval CLI (toy index, tiny model)."""

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import torch.distributed as tdist
import yaml
from transformers import PreTrainedModel

import drbm25.eval.run_eval as run_eval_module
from drbm25.bm25.searcher import Bm25Searcher, SearchResult, TooManyClausesError
from drbm25.checkpoint import BestMeta, TrainState, save_best, save_checkpoint
from drbm25.config import LoraCfg, ScoringMode, load_config
from drbm25.data import DEV_FILE, Example
from drbm25.eval.records import QueryRecord, read_header, read_records
from drbm25.eval.run_eval import main, score_original, score_rewrites
from drbm25.ppo.models import LoraPolicyModels
from drbm25.prompt import build_prompt
from drbm25.rollout.base import RolloutBatch
from drbm25.rollout.hf_generate import HFGenerateEngine
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer

pytestmark = pytest.mark.java

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"
K = 10
REPEATED = "solar solar solar efficiency"


@pytest.fixture(scope="module")
def searcher(toy_index: Path) -> Bm25Searcher:
    return Bm25Searcher(toy_index, k1=0.9, b=0.4, threads=2, timeout_s=30.0)


def _answer(query: str) -> str:
    return f'reasoning</think>\n<answer>\n{{"query": "{query}"}}\n</answer>'


class _TextEngine:
    """Returns fixed response texts (the engine is what scoring does not control)."""

    def __init__(self, texts: Sequence[str]) -> None:
        self._texts = tuple(texts)
        self.greedy_flags: list[bool] = []

    def generate(self, prompts: Sequence[str], *, greedy: bool) -> RolloutBatch:
        del prompts
        self.greedy_flags.append(greedy)
        z = torch.zeros(len(self._texts), 1, dtype=torch.long)
        return RolloutBatch(z, z, z, z, self._texts, torch.zeros(len(self._texts), dtype=torch.bool))

    def sync_weights(self, models: object) -> None:
        del models


def _example(qid: str = "q1", query: str = "solar efficiency", relevant: str = "docB") -> Example:
    return Example(qid=qid, query=query, relevant_ids=frozenset({relevant}))


def test_score_original_uses_query_text(searcher: Bm25Searcher) -> None:
    (record,) = score_original([_example()], searcher, k=K)
    assert record == QueryRecord(
        qid="q1",
        source="original",
        query_text="solar efficiency",
        rewrite_text=None,
        format_error=None,
        terms=("solar", "effici"),
        ndcg_normal=1.0,
        ndcg_count_once=1.0,
    )


def test_format_error_scores_zero_in_both_modes(searcher: Bm25Searcher) -> None:
    engine = _TextEngine(["no tags at all"])
    (record,) = score_rewrites([_example()], engine, searcher, k=K, source="untrained")
    assert record.format_error == "tag_count"
    assert record.rewrite_text == "no tags at all"
    assert record.terms == ()
    assert (record.ndcg_normal, record.ndcg_count_once) == (pytest.approx(0.0), pytest.approx(0.0))
    assert record.source == "untrained"
    assert engine.greedy_flags == [True]


def test_rewrite_scored_under_both_modes_with_repeated_terms(searcher: Bm25Searcher) -> None:
    (record,) = score_rewrites([_example()], _TextEngine([_answer(REPEATED)]), searcher, k=K, source="s")
    assert record.format_error is None
    assert record.rewrite_text == REPEATED
    assert record.terms == ("solar", "solar", "solar", "effici")
    assert record.ndcg_count_once == pytest.approx(1.0)  # docB is first when repeats are ignored
    assert 0.0 < record.ndcg_normal < 1.0  # docA outranks docB when solar counts three times


def test_score_rewrites_records_the_think_length_in_characters(searcher: Bm25Searcher) -> None:
    texts = [
        '  weigh the terms \n</think>\n<answer>\n{"query": "solar"}\n</answer>',
        "no tags at all",
    ]
    records = score_rewrites(
        [_example("q1"), _example("q2")], _TextEngine(texts), searcher, k=K, source="untrained"
    )
    assert [r.think_chars for r in records] == [len("weigh the terms"), None]  # stripped; none on a format error


def test_score_original_has_no_think_length(searcher: Bm25Searcher) -> None:
    records = score_original([_example("q1"), _example("q2")], searcher, k=K)
    assert [r.think_chars for r in records] == [None, None]


def test_score_rewrites_rejects_length_mismatch(searcher: Bm25Searcher) -> None:
    with pytest.raises(ValueError, match="responses"):
        score_rewrites([_example(), _example("q2")], _TextEngine(["x"]), searcher, k=K, source="s")


class _OverflowSearcher:
    """Wraps a real searcher; the 'boom' query exceeds the clause limit in count_once mode."""

    def __init__(self, inner: Bm25Searcher) -> None:
        self._inner = inner

    def analyze(self, text: str) -> list[str]:
        return self._inner.analyze(text)

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult:
        if query == "boom" and mode == "count_once":
            raise TooManyClausesError("too many")
        return self._inner.search("solar efficiency" if query == "boom" else query, mode=mode, k=k)

    def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]:
        return [self.search(q, mode=mode, k=k) for q in queries]


def test_too_many_clauses_scores_zero_in_that_mode_and_is_recorded(searcher: Bm25Searcher) -> None:
    fake = _OverflowSearcher(searcher)
    engine = _TextEngine([_answer("boom"), _answer("solar efficiency")])
    boom, fine = score_rewrites([_example("a"), _example("b")], engine, fake, k=K, source="s")
    assert boom.ndcg_normal == pytest.approx(1.0)
    assert boom.ndcg_count_once == pytest.approx(0.0)
    assert boom.search_error == "too_many_clauses:count_once"
    assert fine.search_error is None
    assert fine.ndcg_count_once == pytest.approx(1.0)


def test_score_original_overflow_is_recorded(searcher: Bm25Searcher) -> None:
    (record,) = score_original([_example(query="boom")], _OverflowSearcher(searcher), k=K)
    assert record.ndcg_count_once == pytest.approx(0.0)
    assert record.search_error == "too_many_clauses:count_once"


# ------------------------------------------------------------------ CLI
@pytest.fixture
def workdir(tmp_path: Path, toy_index: Path) -> Path:
    path = tmp_path / "data" / DEV_FILE
    path.parent.mkdir(parents=True)
    rows = [{"question": f"solar efficiency {i}", "docs_id": ["docB"]} for i in range(5)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return tmp_path


def _argv(workdir: Path, toy_index: Path, out: Path, *extra: str) -> list[str]:
    sets = {
        "retrieval.index_dir": str(toy_index),
        "data.data_dir": str(workdir / "data"),
        "run.output_dir": str(workdir / "out"),
    }
    argv = ["--config", str(SMOKE), "--out", str(out)]
    for key, value in sets.items():
        argv += ["--set", f"{key}={value}"]
    return argv + list(extra)


def test_cli_original_scores_every_dev_query(workdir: Path, toy_index: Path) -> None:
    out = workdir / "orig.jsonl"
    assert main(_argv(workdir, toy_index, out, "--source", "original")) == 0
    records = read_records(out)
    assert [r.qid for r in records] == [f"dev-{i}" for i in range(1, 6)]
    assert {r.source for r in records} == {"original"}
    assert all(r.ndcg_normal == pytest.approx(1.0) and r.ndcg_count_once == pytest.approx(1.0) for r in records)


def test_cli_untrained_runs_tiny_model_greedy(workdir: Path, toy_index: Path) -> None:
    out = workdir / "untrained.jsonl"
    assert main(_argv(workdir, toy_index, out, "--source", "untrained", "--limit", "3")) == 0
    records = read_records(out)
    assert len(records) == 3
    assert {r.source for r in records} == {"untrained"}
    assert all(r.rewrite_text is not None for r in records)  # the tiny random model never produces a valid answer
    assert all(r.format_error is not None and r.ndcg_normal == pytest.approx(0.0) for r in records)


def _write_tiny_checkpoint(workdir: Path, *, break_keys: bool = False, reward_mode: ScoringMode = "normal") -> Path:
    cfg = load_config(SMOKE)
    assert cfg.lora is not None
    lora: LoraCfg = cfg.lora
    models = LoraPolicyModels(build_tiny_qwen2(seed=cfg.run.seed), lora, gradient_checkpointing=False)
    state = models.trainable_state_dict()
    if break_keys:
        state = {f"bad.{k}": v for k, v in state.items()}
    return save_checkpoint(
        workdir / "ckpt_run",
        train_state=TrainState(step=2, data_cursor=8, wandb_run_id=None, reward_mode=reward_mode),
        model=state,
        optimizers={},
        scaler=None,
        keep=1,
    )


@pytest.mark.parametrize("mode", ["normal", "count_once"])
def test_cli_checkpoint_derives_source_from_its_reward_mode(workdir: Path, toy_index: Path, mode: ScoringMode) -> None:
    ckpt = _write_tiny_checkpoint(workdir, reward_mode=mode)
    out = workdir / "trained.jsonl"
    argv = _argv(workdir, toy_index, out, "--source", "checkpoint", "--checkpoint", str(ckpt))
    assert main([*argv, "--limit", "2"]) == 0
    assert {r.source for r in read_records(out)} == {f"trained-{mode}"}


def test_cli_checkpoint_with_wrong_keys_fails(workdir: Path, toy_index: Path) -> None:
    ckpt = _write_tiny_checkpoint(workdir, break_keys=True)
    argv = _argv(workdir, toy_index, workdir / "x.jsonl", "--source", "checkpoint", "--checkpoint", str(ckpt))
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        main([*argv, "--limit", "2"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--source", "checkpoint"],
        ["--source", "original", "--checkpoint", "x"],
        ["--source", "checkpoint", "--checkpoint", "x", "--label", "trained-normal"],
    ],
    ids=["no-checkpoint", "stray-checkpoint", "label-removed"],
)
def test_cli_rejects_inconsistent_source_arguments(workdir: Path, toy_index: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(_argv(workdir, toy_index, workdir / "x.jsonl", *extra))
    assert exc.value.code == 2


# --------------------------------------------------------------------------- full fine-tune (sharded) checkpoints
_CASES_FILE = Path(__file__).resolve().parents[1] / "ppo" / "fsdp_cases.py"


def _load_fsdp_cases() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fsdp_cases", _CASES_FILE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sharded_checkpoint(out: Path) -> dict[str, Any]:
    """Two gloo ranks train a tiny full-layout actor, save it sharded, and record its greedy rewrites."""
    proc = subprocess.run(  # noqa: S603 -- this interpreter and a script in this repo
        [sys.executable, str(_CASES_FILE), "eval_ckpt", str(out)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"), "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return torch.load(out / "result.pt", weights_only=True)


@pytest.mark.slow
def test_full_checkpoint_rewrites_match_the_live_actor(tmp_path: Path) -> None:
    saved = _sharded_checkpoint(tmp_path)
    cases = _load_fsdp_cases()
    cfg = cases.full_cfg()
    assert not tdist.is_initialized()  # a fresh single process, no process group
    prompts = [build_prompt(q) for q in cases.EVAL_QUERIES]
    engine, source = run_eval_module._build_engine(cfg, Path(saved["path"]))
    got = engine.generate(prompts, greedy=True)
    assert source == "trained-count_once"
    assert got.response_ids.tolist() == saved["response_ids"].tolist()
    assert list(got.response_text) == saved["texts"]
    untrained, untrained_source = run_eval_module._build_engine(cfg, None)
    assert untrained_source == "untrained"
    assert untrained.generate(prompts, greedy=True).response_ids.tolist() != saved["response_ids"].tolist()


@pytest.mark.slow
def test_cli_scores_a_full_checkpoint(tmp_path: Path, workdir: Path, toy_index: Path) -> None:
    saved = _sharded_checkpoint(tmp_path)
    full = workdir / "full.yaml"
    raw = yaml.safe_load(SMOKE.read_text())
    raw["model"]["train_mode"] = "full"
    raw["lora"] = None
    full.write_text(yaml.safe_dump(raw))
    out = workdir / "full.jsonl"
    argv = _argv(workdir, toy_index, out, "--source", "checkpoint", "--checkpoint", saved["path"], "--limit", "2")
    argv[argv.index(str(SMOKE))] = str(full)
    assert main(argv) == 0
    assert {r.source for r in read_records(out)} == {"trained-count_once"}


def test_lora_config_refuses_a_sharded_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "step_000001"
    run.mkdir()
    train_state = {"step": 1, "data_cursor": 4, "wandb_run_id": None, "reward_mode": "normal"}
    meta = {"format_version": 1, "layout": "sharded", "train_state": train_state, "rng_python": [], "rng_numpy": {}}
    (run / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="train_mode 'full'"):
        run_eval_module._build_engine(load_config(SMOKE), run)


def test_full_config_refuses_a_lora_checkpoint(workdir: Path) -> None:
    ckpt = _write_tiny_checkpoint(workdir)
    cfg = _load_fsdp_cases().full_cfg()
    with pytest.raises(ValueError, match="not a sharded"):
        run_eval_module._build_engine(cfg, ckpt)


# --------------------------------------------------------------------------- best/ (the selected policy)
_PROMPTS = [build_prompt(q) for q in ("solar panel efficiency", "w4 w9 w12", "heart attack aspirin")]
_SEEDS = {"run_seed": 1234, "split_seed": 1234}


def _push_greedy_away(module: torch.nn.Module, logits_fn: Any, params: list[torch.nn.Parameter]) -> None:
    """A few large LM steps on fixed text, so the greedy rewrites differ from the untrained model's."""
    ids = torch.tensor([[5, 9, 12, 4, 7, 5, 9, 12]])
    mask = torch.ones_like(ids)
    opt = torch.optim.SGD(params, lr=1.0)
    module.train()
    for _ in range(5):
        logits = logits_fn(ids, mask)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    module.eval()


def _meta(layout: str, mode: ScoringMode = "normal") -> BestMeta:
    return BestMeta(
        step=100,
        val_ndcg_normal=0.42,
        val_ndcg_count_once=0.31,
        reward_mode=mode,
        layout="lora" if layout == "lora" else "full",
        config_sha256="ab" * 32,
        seeds=_SEEDS,
        git_sha="f" * 40,
        git_dirty=True,
    )


def _lora_best(run_dir: Path, mode: ScoringMode = "normal") -> tuple[Path, LoraPolicyModels]:
    cfg = load_config(SMOKE)
    assert cfg.lora is not None
    models = LoraPolicyModels(build_tiny_qwen2(seed=cfg.run.seed), cfg.lora, gradient_checkpointing=False)
    _push_greedy_away(models.generation_module(), models.actor_logits, models.actor_parameters())
    return save_best(run_dir, weights=models.actor_state_dict(), meta=_meta("lora", mode)), models


def _full_best(run_dir: Path) -> tuple[Path, PreTrainedModel]:
    actor = build_tiny_qwen2(seed=0)

    def logits(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return actor(input_ids=ids, attention_mask=mask).logits

    _push_greedy_away(actor, logits, list(actor.parameters()))
    return save_best(run_dir, weights=actor.state_dict(), meta=_meta("full", "count_once")), actor


def test_lora_best_rewrites_equal_the_live_actor(tmp_path: Path) -> None:
    cfg = load_config(SMOKE)
    best, live = _lora_best(tmp_path / "run")
    want = HFGenerateEngine(live, build_tiny_tokenizer(), cfg.rollout, torch.device("cpu")).generate(_PROMPTS, greedy=True)
    engine, source = run_eval_module._build_engine(cfg, best)
    got = engine.generate(_PROMPTS, greedy=True)
    assert source == "trained-normal"
    assert got.response_ids.tolist() == want.response_ids.tolist()
    untrained, _ = run_eval_module._build_engine(cfg, None)
    assert untrained.generate(_PROMPTS, greedy=True).response_ids.tolist() != want.response_ids.tolist()


def test_full_best_rewrites_equal_the_live_actor(tmp_path: Path) -> None:
    cfg = _load_fsdp_cases().full_cfg()
    best, live = _full_best(tmp_path / "run")
    frozen = run_eval_module._FrozenActor(live)
    want = HFGenerateEngine(frozen, build_tiny_tokenizer(), cfg.rollout, torch.device("cpu")).generate(
        _PROMPTS, greedy=True
    )
    engine, source = run_eval_module._build_engine(cfg, best)
    got = engine.generate(_PROMPTS, greedy=True)
    assert source == "trained-count_once"
    assert got.response_ids.tolist() == want.response_ids.tolist()
    untrained, _ = run_eval_module._build_engine(cfg, None)
    assert untrained.generate(_PROMPTS, greedy=True).response_ids.tolist() != want.response_ids.tolist()


def test_lora_config_refuses_a_full_best(tmp_path: Path) -> None:
    best, _ = _full_best(tmp_path / "run")
    with pytest.raises(ValueError, match="layout 'full'"):
        run_eval_module._build_engine(load_config(SMOKE), best)


def test_full_config_refuses_a_lora_best(tmp_path: Path) -> None:
    best, _ = _lora_best(tmp_path / "run")
    with pytest.raises(ValueError, match="layout 'lora'"):
        run_eval_module._build_engine(_load_fsdp_cases().full_cfg(), best)


@pytest.mark.parametrize("mode", ["normal", "count_once"])
def test_cli_scores_best_labelled_from_its_meta_and_records_provenance(
    workdir: Path, toy_index: Path, mode: ScoringMode
) -> None:
    best, _ = _lora_best(workdir / "out" / mode, mode)
    out = workdir / "trained.jsonl"
    argv = _argv(workdir, toy_index, out, "--source", "checkpoint", "--checkpoint", str(best), "--limit", "2")
    assert main(argv) == 0
    assert {r.source for r in read_records(out)} == {f"trained-{mode}"}
    header = read_header(out)
    assert header is not None
    assert header["source"] == f"trained-{mode}"
    assert header["checkpoint"] == str(best)
    assert header["best"] == {
        "step": 100,
        "val_ndcg_normal": pytest.approx(0.42),
        "val_ndcg_count_once": pytest.approx(0.31),
        "selection_metric": f"val/ndcg_{mode}",
        "config_sha256": "ab" * 32,
        "seeds": _SEEDS,
        "git": {"sha": "f" * 40, "dirty": True},
    }
    assert header["eval_seed"] == 1234
    assert set(header["git"]) == {"sha", "dirty"}  # the eval code's own version
    assert len(header["eval_config_sha256"]) == 64


def test_cli_best_header_records_the_runs_latest_trained_step(workdir: Path, toy_index: Path) -> None:
    run_dir = workdir / "out" / "normal"
    best, models = _lora_best(run_dir)  # selected at step 100
    state = TrainState(step=150, data_cursor=600, wandb_run_id=None, reward_mode="normal")
    save_checkpoint(run_dir, train_state=state, model=models.trainable_state_dict(), optimizers={}, scaler=None, keep=1)
    out = workdir / "trained.jsonl"
    argv = _argv(workdir, toy_index, out, "--source", "checkpoint", "--checkpoint", str(best), "--limit", "2")
    assert main(argv) == 0
    header = read_header(out)
    assert header is not None
    assert header["trained_steps"] == 150


def test_cli_best_without_a_resume_checkpoint_has_unknown_trained_steps(workdir: Path, toy_index: Path) -> None:
    best, _ = _lora_best(workdir / "out" / "normal")
    out = workdir / "trained.jsonl"
    argv = _argv(workdir, toy_index, out, "--source", "checkpoint", "--checkpoint", str(best), "--limit", "2")
    assert main(argv) == 0
    header = read_header(out)
    assert header is not None
    assert header["trained_steps"] is None


def test_cli_step_checkpoint_header_records_its_own_step(workdir: Path, toy_index: Path) -> None:
    ckpt = _write_tiny_checkpoint(workdir)  # step 2
    out = workdir / "trained.jsonl"
    argv = _argv(workdir, toy_index, out, "--source", "checkpoint", "--checkpoint", str(ckpt), "--limit", "2")
    assert main(argv) == 0
    header = read_header(out)
    assert header is not None
    assert header["trained_steps"] == 2


def test_cli_original_header_has_no_trained_steps(workdir: Path, toy_index: Path) -> None:
    out = workdir / "orig.jsonl"
    assert main(_argv(workdir, toy_index, out, "--source", "original", "--limit", "2")) == 0
    header = read_header(out)
    assert header is not None
    assert header["trained_steps"] is None


def test_cli_original_header_has_no_best(workdir: Path, toy_index: Path) -> None:
    out = workdir / "orig.jsonl"
    assert main(_argv(workdir, toy_index, out, "--source", "original", "--limit", "2")) == 0
    header = read_header(out)
    assert header is not None
    assert (header["source"], header["checkpoint"], header["best"]) == ("original", None, None)
    assert set(header["git"]) == {"sha", "dirty"}
