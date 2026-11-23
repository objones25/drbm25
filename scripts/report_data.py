"""Compute every number and figure used by report/REPORT.md and the slide deck.

Inputs (downloaded beforehand):
  --wandb DIR    <run_id>_history.json / _config.json for the two training runs, and the eval tables of the report run
  --metrics DIR  the metrics/ folder of the objones25/drbm25-rewrites dataset
  --db DIR       the db/ folder of the same dataset
Outputs: report/data/*.json and report/figures/*.png.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from drbm25.eval.sample_db import read_db

RUNS = {"normal": "hunfj08k", "count_once": "yld3obpy"}
COLORS = {"original": "#6b6b6b", "untrained": "#b07aa1", "normal": "#1f6f8b", "count_once": "#e07b39"}
SRC = {"trained-normal": "normal", "trained-count_once": "count_once"}
SMOOTH = 25
TRAIN_KEYS = [
    "reward/mean", "ndcg/train_normal", "ndcg/train_count_once", "ppo/kl_ref_mean", "ppo/kl_ref_max", "ppo/entropy",
    "ppo/value_loss", "ppo/explained_var", "ppo/clipfrac", "ppo/approx_kl", "ppo/policy_loss", "ppo/vf_clipfrac",
    "optim/actor_grad_norm", "optim/actor_clipped", "optim/critic_grad_norm", "optim/critic_clipped",
    "optim/scaler_skipped", "rollout/response_len_mean", "rollout/hit_max_tokens_rate", "reward/format_error_rate",
    "reward/zero_ndcg_rate", "rewrite/n_terms", "rewrite/any_repeat_rate", "rewrite/repeat_ratio",
    "stab/qk_logit_bound_max", "stab/final_norm_out", "sys/gpu_mem_peak_gb", "time/generate", "time/update",
    "val/ndcg_normal", "val/ndcg_count_once", "ppo/values_mean", "ppo/returns_mean",
]  # fmt: skip


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not math.isnan(v) else None


def load_history(path: Path) -> dict[int, dict]:
    """One record per train/step; the last logged record wins (chunk boundaries are logged twice)."""
    out: dict[int, dict] = {}
    for row in json.loads(path.read_text()):
        step = row.get("train/step")
        if isinstance(step, (int, float)):
            out[int(step)] = {**out.get(int(step), {}), **{k: v for k, v in row.items() if _num(v) is not None}}
    return out


def series(hist: dict[int, dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(s, hist[s][key]) for s in sorted(hist) if key in hist[s]]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts], dtype=float)


def smooth(y: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    k = np.ones(w) / w
    pad = np.pad(y, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def window_mean(x, y, lo, hi):
    m = (x >= lo) & (x < hi)
    return float(y[m].mean()) if m.any() else None


def training(wandb_dir: Path, figdir: Path) -> dict:
    hists = {mode: load_history(wandb_dir / f"{rid}_history.json") for mode, rid in RUNS.items()}
    out: dict = {"runs": RUNS, "per_run": {}}
    for mode, h in hists.items():
        r: dict = {"steps": max(h) + 1}
        for key in TRAIN_KEYS:
            x, y = series(h, key)
            if not len(y):
                continue
            r[key] = {
                "first50": window_mean(x, y, 0, 50),
                "last50": window_mean(x, y, max(h) - 49, max(h) + 1),
                "min": float(y.min()),
                "max": float(y.max()),
                "mean": float(y.mean()),
            }
        vx, vy = series(h, f"val/ndcg_{mode}")
        best = int(np.argmax(vy))
        r["val"] = {
            "points": [[int(a), float(b)] for a, b in zip(vx, vy, strict=True)],
            "first": [int(vx[0]), float(vy[0])],
            "best": [int(vx[best]), float(vy[best])],
            "last": [int(vx[-1]), float(vy[-1])],
            "after300_min": float(vy[vx >= 300].min()),
            "after300_max": float(vy[vx >= 300].max()),
        }
        ts = [h[s].get("_runtime") for s in sorted(h)]
        r["n_val_points"] = len(vx)
        r["grad_clipped_frac"] = float(np.mean(series(h, "optim/actor_clipped")[1] > 0.5))
        r["scaler_skipped_total"] = float(series(h, "optim/scaler_skipped")[1].sum())
        r["runtime_h_last_chunk_counter"] = max(t for t in ts if t is not None) / 3600
        # downsampled curves for native charts: every 10th step of the smoothed series
        curves = {}
        for key in ["reward/mean", f"ndcg/train_{mode}", "ppo/kl_ref_mean", "ppo/entropy", "ppo/value_loss",
                    "ppo/explained_var", "rollout/response_len_mean", "ppo/clipfrac", "optim/actor_grad_norm",
                    "rewrite/n_terms", "rewrite/any_repeat_rate"]:  # fmt: skip
            x, y = series(h, key)
            ys = smooth(y)
            curves[key] = [[int(a), float(b)] for a, b in zip(x[::10], ys[::10], strict=True)]
        r["curves"] = curves
        out["per_run"][mode] = r

    # figures
    def fig_curve(key, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for mode, h in hists.items():
            x, y = series(h, key)
            ax.plot(x, y, color=COLORS[mode], alpha=0.15, lw=0.6)
            ax.plot(x, smooth(y), color=COLORS[mode], lw=1.8, label=f"{mode} run")
        ax.set(xlabel="PPO step", ylabel=ylabel, title=title)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=150)
        plt.close(fig)

    fig_curve("reward/mean", "mean reward (training batch)", "train_reward.png", "Training reward")
    fig_curve("ppo/kl_ref_mean", "KL to the reference model", "train_kl.png", "KL to the starting model")
    fig_curve("ppo/entropy", "policy entropy (nats/token)", "train_entropy.png", "Policy entropy")
    fig_curve("rollout/response_len_mean", "response length (tokens)", "train_response_len.png", "Response length")
    fig_curve("ppo/value_loss", "critic value loss", "train_value_loss.png", "Critic value loss")
    fig_curve("ppo/explained_var", "explained variance", "train_explained_var.png", "Critic explained variance")
    fig_curve("ppo/clipfrac", "PPO clip fraction", "train_clipfrac.png", "PPO clip fraction")
    fig_curve("optim/actor_grad_norm", "actor grad norm (before clipping)", "train_grad_norm.png", "Actor gradient norm")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for mode in RUNS:
        x, y = np.array(out["per_run"][mode]["val"]["points"]).T
        ax.plot(x, y, marker="o", ms=3, color=COLORS[mode], label=f"{mode} run (scored with its own reward)")
        bx, by = out["per_run"][mode]["val"]["best"]
        ax.scatter([bx], [by], s=90, facecolors="none", edgecolors=COLORS[mode], lw=2)
    ax.set(xlabel="PPO step", ylabel="validation nDCG@10 (greedy, 500 held-out train queries)",
           title="Validation nDCG every 50 steps (circle = selected checkpoint)")  # fmt: skip
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "val_ndcg.png", dpi=150)
    plt.close(fig)
    return out


def sampling(metrics: Path, db: Path, figdir: Path, wandb_dir: Path) -> tuple[dict, list]:
    comps = json.loads((metrics / "comparisons.json").read_text())
    lex = json.loads((metrics / "lexical.json").read_text())
    variance = json.loads((metrics / "variance.json").read_text())
    qm = pq.read_table(metrics / "query_metrics.parquet").to_pylist()
    sm = pq.read_table(metrics / "sample_metrics.parquet").to_pylist()
    cp = pq.read_table(metrics / "cross_policy.parquet").to_pylist()
    means: dict = defaultdict(dict)
    for name in qm[0]:
        if name in ("source", "qid"):
            continue
        for src in ("original", "untrained", "trained-normal", "trained-count_once"):
            vals = [r[name] for r in qm if r["source"] == src and r[name] is not None]
            if vals:
                means[src][name] = mean(vals)
    classes = {}
    for src in ("untrained", "trained-normal", "trained-count_once"):
        c = Counter(r["rewrite_class"] for r in sm if r["source"] == src and r["rewrite_class"])
        t = sum(c.values())
        classes[src] = {k: c[k] / t for k in ("same", "compression", "expansion", "substitution", "mixed")}
    pair = [c for c in comps if (c["a"], c["b"]) == ("trained-normal", "trained-count_once")]
    tables = {}
    for t in ("ndcg", "paired_ci", "query_stats"):
        f = next(wandb_dir.rglob(f"{t}.table.json"))
        d = json.loads(f.read_text())
        tables[t] = [dict(zip(d["columns"], row, strict=True)) for row in d["data"]]
    out = {
        "means": means,
        "variance": variance,
        "rewrite_class": classes,
        "pair_comparisons": pair,
        "n_pair_metrics": len([c for c in pair if not c["primary"]]),
        "n_pair_sig": len([c for c in pair if not c["primary"] and c["q_by"] is not None and c["q_by"] < 0.05]),
        "untrained_vs_trained": [c for c in comps if c["a"] == "untrained" and c["metric"] in ("ndcg_normal_mean", "distinct_fraction")],
        "c2st": lex["c2st"],
        "js": {k: lex["pairs"][k]["js_divergence"] for k in lex["pairs"]},
        "fightin": {k: {c: {"a": v["top_a"][:15], "b": v["top_b"][:15]} for c, v in lex["pairs"][k]["fightin_words"].items()} for k in lex["pairs"]},
        "cross_policy": {
            "n": len(cp),
            "same_modal_rewrite": mean(r["same_modal_rewrite"] for r in cp),
            "rbo_modal_mean": mean(r["rbo_modal"] for r in cp),
            "rbo_modal_median": median(r["rbo_modal"] for r in cp),
            "rbo_modal_eq1": mean(1.0 if r["rbo_modal"] >= 0.999999 else 0.0 for r in cp),
            "jaccard10_modal_mean": mean(r["jaccard10_modal"] for r in cp),
        },
        "greedy_eval": tables,
    }

    # figures
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ks = [1, 2, 4, 8, 16]
    for src, label in (("untrained", "untrained"), ("trained-normal", "normal"), ("trained-count_once", "count_once")):
        ys = [means[src]["ndcg_normal_mean"]] + [means[src][f"best_of_{k}"] for k in ks[1:]]
        ax.plot(ks, ys, marker="o", color=COLORS[label], label=label)
    ax.axhline(means["original"]["ndcg_normal_mean"], color=COLORS["original"], ls="--", label="raw query")
    ax.set(xscale="log", xticks=ks, xticklabels=[str(k) for k in ks], xlabel="k samples per query",
           ylabel="expected best-of-k nDCG@10", title="Picking the best of k sampled rewrites")  # fmt: skip
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "best_of_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = ["compression", "same", "substitution", "mixed", "expansion"]
    x = np.arange(len(labels))
    for i, (src, label) in enumerate((("untrained", "untrained"), ("trained-normal", "normal"), ("trained-count_once", "count_once"))):
        ax.bar(x + (i - 1) * 0.27, [classes[src][k] for k in labels], 0.27, color=COLORS[label], label=label)
    ax.set_xticks(x, labels)
    ax.set(ylabel="share of rewrites", title="How rewrites change the query's terms")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "rewrite_class.png", dpi=150)
    plt.close(fig)

    rows = read_db(db)
    think = defaultdict(list)
    for r in rows:
        if r.source in SRC and r.think_text is not None:
            think[SRC[r.source]].append(len(r.think_text))
        if r.source == "untrained" and r.think_text is not None:
            think["untrained"].append(len(r.think_text))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label in ("untrained", "normal", "count_once"):
        ax.hist(np.clip(think[label], 0, 1200), bins=60, histtype="step", lw=1.6, density=True, color=COLORS[label], label=label)
    ax.set(xlabel="reasoning length (characters, clipped at 1200)", ylabel="density", title="Reasoning length per sample")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "think_len.png", dpi=150)
    plt.close(fig)
    out["think_len_median"] = {k: median(v) for k, v in think.items()}
    return out, rows


def examples(rows) -> dict:
    by_q: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_q[r.qid][r.source].append(r)

    def modal(rs):
        c = Counter(r.rewrite for r in rs)
        rw = max(c, key=lambda k: (c[k], k is not None))
        return rw, c[rw]

    def pack(qid):
        q = by_q[qid]
        d = {"qid": qid, "query": q["original"][0].query_text, "original_ndcg": q["original"][0].ndcg_normal}
        for src in ("untrained", "trained-normal", "trained-count_once"):
            rw, n = modal(q[src])
            d[src] = {
                "modal_rewrite": rw,
                "modal_count": n,
                "distinct": len({r.rewrite for r in q[src]}),
                "mean_ndcg": mean(r.ndcg_normal for r in q[src]),
                "think_sample0": q[src][0].think_text,
                "rewrite_sample0": q[src][0].rewrite,
                "ndcg_sample0": q[src][0].ndcg_normal,
            }
        return d

    out: dict = {}
    # "how many" kept by normal, replaced by "number" in count_once
    cands = []
    for qid, q in by_q.items():
        n, _ = modal(q["trained-normal"])
        c, _ = modal(q["trained-count_once"])
        if n and c and "how many" in n.lower() and "number" in c.lower() and "how many" not in c.lower():
            cands.append((len(q["original"][0].query_text), qid))
    out["how_many_vs_number"] = [pack(qid) for _, qid in sorted(cands)[:4]]
    out["how_many_vs_number_count"] = len(cands)
    # untrained repetition
    reps = []
    for qid, q in by_q.items():
        r0 = q["untrained"][0]
        if r0.terms and max(Counter(r0.terms).values()) >= 3 and r0.rewrite and len(r0.rewrite) < 90:
            reps.append((r0.ndcg_normal - q["original"][0].ndcg_normal, qid))
    out["untrained_repeat"] = [pack(qid) for _, qid in sorted(reps)[:3]]
    # largest trained gain / loss vs original (normal run, mean over samples)
    deltas = sorted(
        (mean(r.ndcg_normal for r in q["trained-normal"]) - q["original"][0].ndcg_normal, qid) for qid, q in by_q.items()
    )
    out["biggest_losses"] = [pack(qid) for _, qid in deltas[:3]]
    out["biggest_gains"] = [pack(qid) for _, qid in deltas[-3:]]
    out["delta_counts"] = {
        "better": sum(1 for d, _ in deltas if d > 1e-9),
        "worse": sum(1 for d, _ in deltas if d < -1e-9),
        "same": sum(1 for d, _ in deltas if abs(d) <= 1e-9),
    }
    # diversity contrast: normal varies, count_once fixed
    div = []
    for qid, q in by_q.items():
        dn = len({r.rewrite for r in q["trained-normal"]})
        dc = len({r.rewrite for r in q["trained-count_once"]})
        if dc == 1 and dn >= 6 and len(q["original"][0].query_text) < 70:
            div.append((-dn, qid))
    out["diversity_contrast"] = [pack(qid) for _, qid in sorted(div)[:3]]
    return out


def patterns(rows) -> dict:
    """Shares of samples per source showing the patterns seen in the examples (plain string tests)."""
    import re

    tests = {
        "boolean_syntax_in_rewrite": lambda r: bool(r.rewrite) and bool(re.search(r"\b(AND|OR)\b|[()]", r.rewrite)),
        "think_has_digit": lambda r: bool(r.think_text) and bool(re.search(r"\d", r.think_text)),
        "think_starts_broken_down": lambda r: bool(r.think_text) and r.think_text.startswith("The user query can be broken down into key terms"),
        "think_mentions_no_boolean_needed": lambda r: bool(r.think_text) and "does not require the use of Boolean operators" in r.think_text,
        "think_starts_query_is_looking_for": lambda r: bool(r.think_text) and r.think_text.startswith("The query is looking for"),
    }
    out: dict = {}
    for src in ("untrained", "trained-normal", "trained-count_once"):
        rs = [r for r in rows if r.source == src]
        out[src] = {name: sum(1 for r in rs if t(r)) / len(rs) for name, t in tests.items()}
        out[src]["n"] = len(rs)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("report"))
    ns = p.parse_args()
    (ns.out / "data").mkdir(parents=True, exist_ok=True)
    (ns.out / "figures").mkdir(parents=True, exist_ok=True)
    tr = training(ns.wandb, ns.out / "figures")
    sa, rows = sampling(ns.metrics, ns.db, ns.out / "figures", ns.wandb)
    ex = examples(rows)
    sa["patterns"] = patterns(rows)
    for name, obj in (("training", tr), ("sampling", sa), ("examples", ex)):
        (ns.out / "data" / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str) + "\n")
    print("wrote", ns.out)


if __name__ == "__main__":
    main()
