"""Scoring original queries and rewrites, and the run_eval CLI (toy index, tiny model)."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from drbm25.bm25.searcher import Bm25Searcher, SearchResult, TooManyClausesError
from drbm25.checkpoint import TrainState, save_checkpoint
from drbm25.config import LoraCfg, ScoringMode, load_config
from drbm25.data import DEV_FILE, Example
from drbm25.eval.records import QueryRecord, read_records
from drbm25.eval.run_eval import main, score_original, score_rewrites
from drbm25.ppo.models import LoraPolicyModels
from drbm25.rollout.base import RolloutBatch
from drbm25.tiny import build_tiny_qwen2

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


def _write_tiny_checkpoint(workdir: Path, *, break_keys: bool = False) -> Path:
    cfg = load_config(SMOKE)
    assert cfg.lora is not None
    lora: LoraCfg = cfg.lora
    models = LoraPolicyModels(build_tiny_qwen2(seed=cfg.run.seed), lora, gradient_checkpointing=False)
    state = models.trainable_state_dict()
    if break_keys:
        state = {f"bad.{k}": v for k, v in state.items()}
    return save_checkpoint(
        workdir / "ckpt_run",
        train_state=TrainState(step=2, data_cursor=8, wandb_run_id=None),
        model=state,
        optimizers={},
        scaler=None,
        keep=1,
    )


def test_cli_checkpoint_loads_trainable_state_and_uses_label(workdir: Path, toy_index: Path) -> None:
    ckpt = _write_tiny_checkpoint(workdir)
    out = workdir / "trained.jsonl"
    argv = _argv(workdir, toy_index, out, "--source", "checkpoint", "--checkpoint", str(ckpt))
    assert main([*argv, "--label", "trained-normal", "--limit", "2"]) == 0
    assert {r.source for r in read_records(out)} == {"trained-normal"}


def test_cli_checkpoint_with_wrong_keys_fails(workdir: Path, toy_index: Path) -> None:
    ckpt = _write_tiny_checkpoint(workdir, break_keys=True)
    argv = _argv(workdir, toy_index, workdir / "x.jsonl", "--source", "checkpoint", "--checkpoint", str(ckpt))
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        main([*argv, "--label", "trained-normal", "--limit", "2"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--source", "checkpoint", "--label", "trained-normal"],
        ["--source", "checkpoint", "--checkpoint", "x"],
        ["--source", "original", "--checkpoint", "x"],
        ["--source", "original", "--label", "trained-normal"],
    ],
    ids=["no-checkpoint", "no-label", "stray-checkpoint", "stray-label"],
)
def test_cli_rejects_inconsistent_source_arguments(workdir: Path, toy_index: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(_argv(workdir, toy_index, workdir / "x.jsonl", *extra))
    assert exc.value.code == 2
