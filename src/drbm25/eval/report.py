"""Exp 1 / Exp 2 report from per-source JSON-lines files.

    python -m drbm25.eval.report --inputs A.jsonl B.jsonl ... --out report.md
        [--n-resamples N] [--wandb --config PATH]

Each input file holds one source (see ``drbm25.eval.records``). All sources must cover the
same qids. Output: nDCG@10 per source under both scorers, paired bootstrap CIs, and query
statistics per source.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import wandb
from drbm25.checkpoint import complete_steps
from drbm25.config import load_config
from drbm25.eval.records import (
    SOURCE_ORIGINAL,
    SOURCE_TRAINED_COUNT_ONCE,
    SOURCE_TRAINED_NORMAL,
    SOURCE_UNTRAINED,
    SOURCES,
    QueryRecord,
    read_records,
)
from drbm25.eval.stats import PairedCI, paired_bootstrap_ci, query_stats
from drbm25.obs import configure
from drbm25.obs.runs import open_run

logger = logging.getLogger(__name__)

_NORMAL = "nDCG@10 normal"
_COUNT_ONCE = "nDCG@10 count_once"
_STAT_NAMES = ("mean_n_terms", "mean_n_unique_terms", "repeat_ratio", "any_repeat_rate", "format_error_rate")
_STAT_FORMATS = ("{:.2f}", "{:.2f}", "{:.3f}", "{:.3f}", "{:.3f}")


def common_step(run_dir_a: Path, run_dir_b: Path) -> int:
    """The largest step with a complete checkpoint in both run directories."""
    shared = set(complete_steps(run_dir_a)) & set(complete_steps(run_dir_b))
    if not shared:
        raise ValueError(f"no common step between {run_dir_a} and {run_dir_b}")
    return max(shared)


@dataclass(frozen=True)
class Report:
    n_queries: int
    ndcg: list[tuple[str, float, float]]  # source, mean normal, mean count_once
    paired_ci: list[tuple[str, str, PairedCI]]  # comparison, scorer, CI of (b - a)
    stats: list[tuple[str, dict[str, float]]]


def _align(sources: Mapping[str, Sequence[QueryRecord]]) -> dict[str, list[QueryRecord]]:
    """Each source's records ordered by qid; every source must hold the same qid set."""
    if not sources:
        raise ValueError("no inputs")
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        raise ValueError(f"unknown source {unknown}; expected a subset of {SOURCES}")
    aligned: dict[str, list[QueryRecord]] = {}
    for name in SOURCES:
        if name not in sources:
            continue
        qids = [r.qid for r in sources[name]]
        if len(set(qids)) != len(qids):
            raise ValueError(f"source {name} has a duplicate qid")
        if not qids:
            raise ValueError(f"source {name} has no records")
        aligned[name] = sorted(sources[name], key=lambda r: r.qid)
    reference_name, reference = next(iter(aligned.items()))
    reference_qids = [r.qid for r in reference]
    for name, records in aligned.items():
        if [r.qid for r in records] != reference_qids:
            raise ValueError(f"qid sets differ between {reference_name} and {name}")
    return aligned


def _scores(records: Sequence[QueryRecord], scorer: str) -> np.ndarray:
    return np.array([r.ndcg_normal if scorer == _NORMAL else r.ndcg_count_once for r in records])


def build_report_data(
    sources: Mapping[str, Sequence[QueryRecord]], *, n_resamples: int = 10_000, seed: int = 0
) -> Report:
    aligned = _align(sources)

    def ci(a: np.ndarray, b: np.ndarray) -> PairedCI:
        return paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)

    rows: list[tuple[str, str, PairedCI]] = []
    if SOURCE_TRAINED_NORMAL in aligned and SOURCE_TRAINED_COUNT_ONCE in aligned:
        for scorer in (_NORMAL, _COUNT_ONCE):
            a = _scores(aligned[SOURCE_TRAINED_NORMAL], scorer)
            b = _scores(aligned[SOURCE_TRAINED_COUNT_ONCE], scorer)
            rows.append((f"{SOURCE_TRAINED_COUNT_ONCE} - {SOURCE_TRAINED_NORMAL}", scorer, ci(a, b)))
    for name in (SOURCE_TRAINED_NORMAL, SOURCE_TRAINED_COUNT_ONCE):
        if name in aligned:
            gap = ci(_scores(aligned[name], _COUNT_ONCE), _scores(aligned[name], _NORMAL))
            rows.append((f"{name}: normal - count_once", "nDCG@10", gap))
    if SOURCE_UNTRAINED in aligned and SOURCE_ORIGINAL in aligned:
        for scorer in (_NORMAL, _COUNT_ONCE):
            a, b = _scores(aligned[SOURCE_ORIGINAL], scorer), _scores(aligned[SOURCE_UNTRAINED], scorer)
            rows.append((f"{SOURCE_UNTRAINED} - {SOURCE_ORIGINAL}", scorer, ci(a, b)))
    return Report(
        n_queries=len(next(iter(aligned.values()))),
        ndcg=[
            (name, float(_scores(r, _NORMAL).mean()), float(_scores(r, _COUNT_ONCE).mean()))
            for name, r in aligned.items()
        ],
        paired_ci=rows,
        stats=[(name, query_stats(r)) for name, r in aligned.items()],
    )


def render_markdown(report: Report) -> str:
    lines = [
        f"Queries: {report.n_queries}",
        "",
        f"| source | {_NORMAL} | {_COUNT_ONCE} |",
        "|---|---|---|",
        *(f"| {name} | {n:.4f} | {c:.4f} |" for name, n, c in report.ndcg),
        "",
    ]
    if report.paired_ci:
        lines += [
            "Paired bootstrap 95% CI of the mean per-query difference, as written (b - a).",
            "",
            "| comparison | scorer | mean diff | CI low | CI high |",
            "|---|---|---|---|---|",
            *(
                f"| {name} | {scorer} | {ci.mean_diff:+.4f} | {ci.low:+.4f} | {ci.high:+.4f} |"
                for name, scorer, ci in report.paired_ci
            ),
            "",
        ]
    lines += [
        f"| source | {' | '.join(_STAT_NAMES)} |",
        "|---|" + "---|" * len(_STAT_NAMES),
        *(
            f"| {name} | {' | '.join(fmt.format(values[key]) for key, fmt in zip(_STAT_NAMES, _STAT_FORMATS, strict=True))} |"
            for name, values in report.stats
        ),
        "",
    ]
    return "\n".join(lines)


def build_report(sources: Mapping[str, Sequence[QueryRecord]], *, n_resamples: int = 10_000, seed: int = 0) -> str:
    return render_markdown(build_report_data(sources, n_resamples=n_resamples, seed=seed))


def _load_inputs(paths: Sequence[Path]) -> dict[str, list[QueryRecord]]:
    sources: dict[str, list[QueryRecord]] = {}
    for path in paths:
        records = read_records(path)
        names = {r.source for r in records}
        if len(names) > 1:
            raise ValueError(f"{path} holds more than one source: {sorted(names)}")
        if not names:
            raise ValueError(f"{path} has no records")
        (name,) = names
        if name in sources:
            raise ValueError(f"source {name} appears in more than one input")
        sources[name] = records
    return sources


def _log_to_wandb(
    cfg_path: Path, overrides: Mapping[str, str], report: Report, inputs: Sequence[Path], n_resamples: int
) -> None:
    cfg = load_config(cfg_path, dict(overrides))
    extra: dict[str, object] = {
        "inputs": [str(p) for p in inputs],
        "n_queries": report.n_queries,
        "n_resamples": n_resamples,
    }
    with open_run(
        cfg, run_name="report", reward_mode="normal", job_type="report", resume_id=None, extra_config=extra
    ) as run:
        run.log(
            {
                "report/ndcg": wandb.Table(
                    columns=["source", "ndcg10_normal", "ndcg10_count_once"],
                    data=[[n, a, b] for n, a, b in report.ndcg],
                ),
                "report/paired_ci": wandb.Table(
                    columns=["comparison", "scorer", "mean_diff", "ci_low", "ci_high"],
                    data=[[n, s, ci.mean_diff, ci.low, ci.high] for n, s, ci in report.paired_ci],
                ),
                "report/query_stats": wandb.Table(
                    columns=["source", *_STAT_NAMES],
                    data=[[n, *(v[key] for key in _STAT_NAMES)] for n, v in report.stats],
                ),
            }
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m drbm25.eval.report", description="Exp 1 / Exp 2 report.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--wandb", action="store_true", help="also log the tables to a W&B run (job_type report)")
    parser.add_argument("--config", type=Path, default=None, help="config for the W&B project (with --wandb)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ns = parser.parse_args(argv)
    if ns.wandb and ns.config is None:
        parser.error("--wandb needs --config (it names the W&B project)")
    overrides: dict[str, str] = {}
    for item in ns.overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--set expects key=value, got {item!r}")
        overrides[key] = value
    configure(level="INFO", fmt="json")
    report = build_report_data(_load_inputs(ns.inputs), n_resamples=ns.n_resamples)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(render_markdown(report), encoding="utf-8")
    logger.info("wrote %s", ns.out)
    if ns.wandb:
        _log_to_wandb(ns.config, overrides, report, ns.inputs, ns.n_resamples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
