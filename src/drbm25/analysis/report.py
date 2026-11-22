"""Log the behavior analysis to W&B and build a W&B Report from it.

CLI: ``python -m drbm25.analysis.report --metrics DIR --db DIR --config PATH [--work DIR] [--browse-rows N]
[--no-report] [--set key=value ...]``.

One W&B run (``rewrite-analysis``, job_type ``analysis``) gets: per-source means as summary scalars, the
per-query metrics, comparisons, Fightin' Words, classifier tests and variance components as Tables, a browsable
sample of rows (query, reasoning, rewrite, scores), matplotlib figures as Images, and the database plus metrics as
a ``rewrite-db`` dataset artifact. The run is the source of truth: the Report (wandb-workspaces Reports API, in
public preview) only arranges its panels, and a failure to build it is logged, not raised.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import wandb

from drbm25.analysis.pipeline import BEST_OF_K, MODEL_SOURCES, PRIMARY_PAIR
from drbm25.config import Config, load_config
from drbm25.eval.records import SOURCE_ORIGINAL
from drbm25.eval.sample_db import read_db
from drbm25.obs import configure
from drbm25.obs.runs import open_run

logger = logging.getLogger(__name__)

RUN_NAME: Final = "rewrite-analysis"
ALL_SOURCES: Final = (SOURCE_ORIGINAL, *MODEL_SOURCES)
REPORT_TITLE: Final = "Query rewrite behavior: untrained vs normal vs count-once"
DIST_METRICS: Final = (
    "mean_n_terms",
    "mean_think_chars",
    "mean_retention",
    "mean_duplicate_rate",
    "mean_embed_sim",
    "mean_idf_mean",
    "mean_rbo_vs_original",
)


# ---------------------------------------------------------------- inputs


def load_outputs(metrics_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("sample_metrics", "query_metrics", "cross_policy"):
        out[name] = pq.read_table(metrics_dir / f"{name}.parquet").to_pylist()
    for name in ("comparisons", "lexical", "variance"):
        out[name] = json.loads((metrics_dir / f"{name}.json").read_text(encoding="utf-8"))
    return out


def _by_source(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None:
            values[r["source"]].append(float(v))
    return values


def source_means(per_query: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean over queries of every numeric per-query metric, per source."""
    names = [k for k in per_query[0] if k not in ("source", "qid")] if per_query else []
    out: dict[str, dict[str, float]] = {}
    for name in names:
        for source, values in _by_source(per_query, name).items():
            out.setdefault(source, {})[name] = mean(values)
    return out


# ---------------------------------------------------------------- summary text


def _fmt_cmp(c: Mapping[str, Any]) -> str:
    return (
        f"{c['b']} - {c['a']}: {c['mean_diff']:+.4f} (95% CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}], "
        f"Wilcoxon p = {c['wilcoxon_p']:.3g}, {c['n_queries']} queries)"
    )


def summary_markdown(outputs: Mapping[str, Any]) -> str:
    means = source_means(outputs["query_metrics"])
    lines = [
        "| source | nDCG@10 normal | nDCG@10 count-once | format errors | distinct rewrites |",
        "|---|---|---|---|---|",
    ]
    for source in ALL_SOURCES:
        m = means.get(source)
        if m is None:
            continue
        lines.append(
            f"| {source} | {m['ndcg_normal_mean']:.4f} | {m['ndcg_count_once_mean']:.4f} | "
            f"{m.get('mean_is_format_error', 0.0):.3f} | {m.get('distinct_fraction', float('nan')):.3f} |"
        )
    lines += ["", "**Primary comparison** (per-query mean over samples, paired query-level bootstrap):", ""]
    lines += [f"- {c['metric']}: {_fmt_cmp(c)}" for c in outputs["comparisons"] if c["primary"]]
    significant = sorted(
        (c for c in outputs["comparisons"] if not c["primary"] and c["q_by"] is not None and c["q_by"] < 0.05),
        key=lambda c: c["q_by"],
    )
    pair = [c for c in significant if (c["a"], c["b"]) == PRIMARY_PAIR]
    lines += ["", f"**Normal vs count-once, other metrics** (BY-adjusted q < 0.05): {len(pair)} of "]
    lines[-1] += f"{sum(1 for c in outputs['comparisons'] if (c['a'], c['b']) == PRIMARY_PAIR and not c['primary'])}"
    lines += [f"- {c['metric']}: {_fmt_cmp(c)}, q = {c['q_by']:.3g}" for c in pair[:15]]
    c2st = outputs["lexical"].get("c2st", {})
    if c2st:
        lines += ["", "**Can a classifier tell normal from count-once?** (sample 0 of each query, grouped CV)"]
        lines += [
            f"- {kind}: accuracy {r['accuracy']:.3f}, permutation p = {r['p_value']:.3g}" for kind, r in c2st.items()
        ]
    fw = outputs["lexical"]["pairs"].get(f"{PRIMARY_PAIR[0]}|{PRIMARY_PAIR[1]}", {}).get("fightin_words", {})
    uni = fw.get("rewrite_1")
    if uni and uni["top_a"]:
        lines += [
            "",
            f"**Words {PRIMARY_PAIR[0]} uses more:** " + ", ".join(f"{w} ({z:.1f})" for w, z in uni["top_a"][:10]),
            f"**Words {PRIMARY_PAIR[1]} uses more:** " + ", ".join(f"{w} ({z:.1f})" for w, z in uni["top_b"][:10]),
        ]
    return "\n".join(lines)


METHODS_MARKDOWN: Final = """\
- **Data:** every dev query sampled N times per policy at the training temperature (0.6, top-p 1, no top-k) with \
the training prompt; the same seed per query across policies. Responses are parsed and scored with the training \
parser and both BM25 scorers. `original` is the raw query (one row per query).
- **Unit of analysis:** the query. Each metric is averaged over a query's samples first; policies are compared on \
those per-query values with a paired bootstrap that resamples queries (10,000 resamples) and a Wilcoxon \
signed-rank test.
- **Multiple comparisons:** nDCG@10 of normal vs count-once is the primary comparison; all other comparisons are \
Benjamini-Yekutieli adjusted (valid under dependence between metrics).
- **Metrics:** term retention, Jaccard and added/dropped terms vs the original query; repetition; IDF of rewrite \
and added terms; RBO (p = 0.9) and Jaccard@10 of the result list vs the original query's; embedding cosine \
(all-MiniLM-L6-v2); distinct rewrites, Vendi score (term-set Jaccard kernel), expected best-of-k nDCG and pairwise \
RBO across a query's samples; Fightin' Words (informative Dirichlet prior from pooled counts, a0 = 1000), \
Jensen-Shannon divergence, and a classifier two-sample test with query-grouped folds and within-query label \
permutation; between-/within-query variance components of nDCG.
"""


# ---------------------------------------------------------------- figures


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def make_figures(outputs: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    per_query = outputs["query_metrics"]
    figures: dict[str, Path] = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for source, values in sorted(_by_source(per_query, "ndcg_normal_mean").items()):
        xs = np.sort(values)
        ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", label=source)
    ax.set(xlabel="per-query mean nDCG@10 (standard BM25)", ylabel="fraction of queries", title="nDCG@10 ECDF")
    ax.legend()
    figures["figures/ndcg_ecdf"] = _save(fig, out_dir / "ndcg_ecdf.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    means = source_means(per_query)
    for source in MODEL_SOURCES:
        if source in means:
            ks = [k for k in BEST_OF_K if means[source].get(f"best_of_{k}") is not None]
            ax.plot(ks, [means[source][f"best_of_{k}"] for k in ks], marker="o", label=source)
    ax.set(xscale="log", xlabel="k", ylabel="expected best-of-k nDCG@10", title="Best of k samples")
    ax.legend()
    figures["figures/best_of_k"] = _save(fig, out_dir / "best_of_k.png")

    for metric in DIST_METRICS:
        values = _by_source(per_query, metric)
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for source, vs in sorted(values.items()):
            ax.hist(vs, bins=40, histtype="step", label=source, density=True)
        ax.set(xlabel=f"per-query {metric}", ylabel="density", title=metric)
        ax.legend()
        figures[f"figures/{metric}"] = _save(fig, out_dir / f"{metric}.png")

    classes: dict[str, Counter[str]] = defaultdict(Counter)
    for r in outputs["sample_metrics"]:
        if r["source"] in MODEL_SOURCES and r.get("rewrite_class"):
            classes[r["source"]][r["rewrite_class"]] += 1
    if classes:
        labels = ["same", "expansion", "compression", "substitution", "mixed"]
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / len(classes)
        for i, (source, counts) in enumerate(sorted(classes.items())):
            total = sum(counts.values())
            ax.bar(np.arange(len(labels)) + i * width, [counts[k] / total for k in labels], width, label=source)
        ax.set_xticks(np.arange(len(labels)) + width * (len(classes) - 1) / 2, labels)
        ax.set(ylabel="share of rewrites", title="Rewrite type vs the original query")
        ax.legend()
        figures["figures/rewrite_class"] = _save(fig, out_dir / "rewrite_class.png")

    fw = outputs["lexical"]["pairs"].get(f"{PRIMARY_PAIR[0]}|{PRIMARY_PAIR[1]}", {}).get("fightin_words", {})
    for corpus in ("rewrite_1", "think_1"):
        entry = fw.get(corpus)
        if not entry or not entry["top_a"]:
            continue
        top = entry["top_a"][:15][::-1] + entry["top_b"][:15]
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.barh([w for w, _ in top], [z for _, z in top], color=["tab:blue" if z > 0 else "tab:orange" for _, z in top])
        ax.set(xlabel=f"z (+: {PRIMARY_PAIR[0]}, -: {PRIMARY_PAIR[1]})", title=f"Fightin' Words, {corpus}")
        figures[f"figures/fightin_{corpus}"] = _save(fig, out_dir / f"fightin_{corpus}.png")

    rbo = [r["rbo_modal"] for r in outputs["cross_policy"]]
    if rbo:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(rbo, bins=40)
        ax.set(
            xlabel="RBO of the two policies' modal-rewrite result lists", ylabel="queries", title="Normal vs count-once"
        )
        figures["figures/cross_policy_rbo"] = _save(fig, out_dir / "cross_policy_rbo.png")
    return figures


# ---------------------------------------------------------------- W&B run


def _table(rows: Sequence[Mapping[str, Any]]) -> wandb.Table:
    columns: list[Any] = list(rows[0]) if rows else ["empty"]
    return wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows])


def _browse_rows(db: Path, n: int, seed: int = 0) -> list[dict[str, Any]]:
    rows = read_db(db)
    rng = random.Random(seed)  # noqa: S311 -- reproducible row sampling for display, not security
    picked: list[dict[str, Any]] = []
    per_source = max(1, n // len(MODEL_SOURCES))
    for source in MODEL_SOURCES:
        pool = [r for r in rows if r.source == source]
        for r in rng.sample(pool, min(per_source, len(pool))):
            picked.append(
                {
                    "source": r.source,
                    "qid": r.qid,
                    "sample": r.sample_index,
                    "query": r.query_text,
                    "reasoning": r.think_text,
                    "rewrite": r.rewrite,
                    "format_error": r.format_error,
                    "ndcg_normal": r.ndcg_normal,
                    "ndcg_count_once": r.ndcg_count_once,
                }
            )
    return picked[:n]


def _lexical_rows(lexical: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair, entry in lexical["pairs"].items():
        a, b = pair.split("|")
        for corpus, fw in entry["fightin_words"].items():
            rows += [{"pair": pair, "corpus": corpus, "favors": a, "term": w, "z": z} for w, z in fw["top_a"]]
            rows += [{"pair": pair, "corpus": corpus, "favors": b, "term": w, "z": z} for w, z in fw["top_b"]]
    return rows


def log_analysis(cfg: Config, metrics_dir: Path, db: Path, *, work: Path, browse_rows: int) -> str:
    """Log tables, figures, scalars and the dataset artifact to one W&B run; returns ``entity/project/id``."""
    outputs = load_outputs(metrics_dir)
    figures = make_figures(outputs, work / "figures")
    extra = {"metrics_dir": str(metrics_dir), "db": str(db), "browse_rows": browse_rows}
    with open_run(
        cfg, run_name=RUN_NAME, reward_mode="normal", job_type="analysis", resume_id=None, extra_config=extra
    ) as run:
        for source, metrics in source_means(outputs["query_metrics"]).items():
            for name, value in metrics.items():
                run.summary[f"summary/{source}/{name}"] = value
        for c in outputs["comparisons"]:
            if c["primary"]:
                for key in ("mean_diff", "ci_low", "ci_high", "wilcoxon_p"):
                    run.summary[f"primary/{c['metric']}/{key}"] = c[key]
        variance = [{"source": s, **v} for s, v in outputs["variance"].items()]
        c2st = [
            {"text": k, **{n: v for n, v in r.items() if n in ("accuracy", "p_value")}}
            for k, r in outputs["lexical"]["c2st"].items()
        ]
        run.log(
            {
                "tables/query_metrics": _table(outputs["query_metrics"]),
                "tables/comparisons": _table(outputs["comparisons"]),
                "tables/cross_policy": _table(outputs["cross_policy"]),
                "tables/fightin_words": _table(_lexical_rows(outputs["lexical"])),
                "tables/c2st": _table(c2st),
                "tables/variance": _table(variance),
                "tables/browse": _table(_browse_rows(db, browse_rows)),
                **{key: wandb.Image(str(path)) for key, path in figures.items()},
            }
        )
        artifact = wandb.Artifact(name="rewrite-db", type="dataset")
        artifact.add_dir(str(db), name="db")
        artifact.add_dir(str(metrics_dir), name="metrics")
        run.log_artifact(artifact)
        return f"{run.entity}/{run.project}/{run.id}"


# ---------------------------------------------------------------- report


def build_report(entity: str, project: str, run_name: str, outputs: Mapping[str, Any], *, wr: Any = None) -> str:
    """Save a W&B Report arranging the analysis run's panels; returns its URL."""
    if wr is None:
        import wandb_workspaces.reports.v2 as wr  # optional dependency, public-preview API

    def grid(*panels: Any) -> Any:
        return wr.PanelGrid(runsets=[wr.Runset(entity=entity, project=project, query=run_name)], panels=list(panels))

    def images(*keys: str) -> Any:
        return wr.MediaBrowser(media_keys=list(keys))

    blocks = [
        wr.H2(text="Summary"),
        wr.MarkdownBlock(text=summary_markdown(outputs)),
        wr.H2(text="Retrieval quality"),
        wr.P(text="Per-query mean nDCG@10 over samples, and how much picking the best of k samples would gain."),
        grid(
            images("figures/ndcg_ecdf", "figures/best_of_k"), wr.WeavePanelSummaryTable(table_name="tables/comparisons")
        ),
        wr.H2(text="Rewrite behavior"),
        wr.P(
            text="How rewrites relate to the original query: kept/added terms, repetition, specificity, retrieved documents."
        ),
        grid(
            images(
                "figures/rewrite_class",
                "figures/mean_retention",
                "figures/mean_duplicate_rate",
                "figures/mean_idf_mean",
            ),
            images(
                "figures/mean_n_terms",
                "figures/mean_embed_sim",
                "figures/mean_rbo_vs_original",
                "figures/cross_policy_rbo",
            ),
        ),
        wr.H2(text="Diversity"),
        wr.P(
            text="Within-query diversity across samples (distinct rewrites, Vendi score, pairwise RBO) and variance components."
        ),
        grid(
            wr.WeavePanelSummaryTable(table_name="tables/query_metrics"),
            wr.WeavePanelSummaryTable(table_name="tables/variance"),
        ),
        wr.H2(text="Vocabulary"),
        wr.P(
            text="Words each policy over-uses (Fightin' Words z-scores) and whether a classifier can tell them apart."
        ),
        grid(
            images("figures/fightin_rewrite_1"),
            wr.WeavePanelSummaryTable(table_name="tables/fightin_words"),
            wr.WeavePanelSummaryTable(table_name="tables/c2st"),
        ),
        wr.H2(text="Reasoning"),
        wr.P(text="Reasoning length and vocabulary, and a browsable sample of query, reasoning and rewrite."),
        grid(
            images("figures/mean_think_chars", "figures/fightin_think_1"),
            wr.WeavePanelSummaryTable(table_name="tables/browse"),
        ),
        wr.H2(text="Methods"),
        wr.MarkdownBlock(text=METHODS_MARKDOWN),
    ]
    report = wr.Report(
        project=project,
        entity=entity,
        title=REPORT_TITLE,
        description="Sampled rewrites of every dev query.",
        blocks=blocks,
    )
    report.save()
    return str(report.url)


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log the rewrite behavior analysis to W&B and build a report.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="config naming the W&B project")
    parser.add_argument("--work", type=Path, default=None, help="scratch dir for figures (default: <metrics>/report)")
    parser.add_argument("--browse-rows", type=int, default=5000)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ns = parser.parse_args(argv)
    overrides = dict(item.partition("=")[::2] for item in ns.overrides)
    configure(level="INFO", fmt="json")
    cfg = load_config(ns.config, overrides)
    run_path = log_analysis(cfg, ns.metrics, ns.db, work=ns.work or ns.metrics / "report", browse_rows=ns.browse_rows)
    logger.info("analysis run: %s", run_path)
    if not ns.no_report:
        entity, project, _ = run_path.split("/")
        try:
            url = build_report(entity, project, RUN_NAME, load_outputs(ns.metrics))
        except Exception:  # the run already holds every result; the preview Reports API must not fail the job
            logger.exception("building the W&B Report failed; the analysis run %s has all results", run_path)
        else:
            logger.info("W&B report: %s", url)
            (ns.metrics / "REPORT_URL").write_text(url + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
