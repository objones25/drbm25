"""Slide-sized chart images for report/drbm25_results.pptx, from report/data/*.json (see scripts/report_data.py)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

REPORT = Path("report")
T = json.loads((REPORT / "data/training.json").read_text())
S = json.loads((REPORT / "data/sampling.json").read_text())
OUT = REPORT / "figures"
HEX = {"normal": "#1F6F8B", "once": "#E07B39", "untrained": "#B07AA1", "raw": "#6B6B6B", "ink": "#1B2631", "grid": "#D9E1E6"}
plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "axes.edgecolor": "#8A9AA5", "axes.labelcolor": "#3D4A52",
    "xtick.color": "#3D4A52", "ytick.color": "#3D4A52", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": HEX["ink"], "axes.titlelocation": "left",
})  # fmt: skip


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def grid(ax):
    ax.grid(axis="y", color=HEX["grid"], lw=0.8)
    ax.set_axisbelow(True)


def curve_pair(key, title, ylabel, name, size, fmt=None):
    fig, ax = plt.subplots(figsize=size)
    for mode, color, label in (("normal", HEX["normal"], "normal reward"), ("count_once", HEX["once"], "count-once reward")):
        pts = T["per_run"][mode]["curves"][key]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2.2, label=label)
    ax.set(title=title, xlabel="PPO step", ylabel=ylabel)
    if fmt:
        ax.yaxis.set_major_formatter(fmt)
    grid(ax)
    ax.legend(frameon=False, loc="best")
    save(fig, name)


def bars(labels, values, colors, title, name, size, pct=False, ylim=None):
    fig, ax = plt.subplots(figsize=size)
    b = ax.bar(labels, values, color=colors, width=0.62)
    for rect, v in zip(b, values, strict=True):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{100 * v:.1f}%" if pct else f"{v:.3f}", ha="center", va="bottom", fontsize=13, color=HEX["ink"])
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    grid(ax)
    save(fig, name)


def main() -> None:
    wide, half = (8.2, 5.4), (6.0, 3.0)
    curve_pair("reward/mean", "Mean reward per batch (25-step moving average)", "reward", "deck_reward.png", wide)
    curve_pair("ppo/entropy", "Policy entropy per token", "entropy", "deck_entropy.png", half)
    curve_pair("ppo/kl_ref_mean", "KL divergence from the starting model", "KL", "deck_kl.png", half)
    curve_pair("rewrite/any_repeat_rate", "Training rewrites with a repeated term (25-step average)", "share of batch", "deck_repeat_train.png", wide, fmt=PercentFormatter(1.0, decimals=0))
    curve_pair("ppo/explained_var", "Critic explained variance (25-step average)", "explained variance", "deck_explained_var.png", wide)

    fig, ax = plt.subplots(figsize=wide)
    for mode, color, label in (("normal", HEX["normal"], "normal"), ("count_once", HEX["once"], "count-once")):
        v = T["per_run"][mode]["val"]
        ax.plot([p[0] for p in v["points"]], [p[1] for p in v["points"]], color=color, lw=2, marker="o", ms=4, label=label)
        ax.scatter([v["best"][0]], [v["best"][1]], s=140, facecolors="none", edgecolors=color, lw=2)
    ax.set(title="Validation nDCG@10 (circle = selected checkpoint)", xlabel="PPO step", ylabel="greedy nDCG@10, 500 queries", ylim=(0.32, 0.385))
    grid(ax)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "deck_val.png")

    g = {r["source"]: r for r in S["greedy_eval"]["ndcg"]}
    srcs = ["original", "untrained", "trained-normal", "trained-count_once"]
    labels = ["raw query", "untrained", "normal", "count-once"]
    colors = [HEX["raw"], HEX["untrained"], HEX["normal"], HEX["once"]]
    bars(labels, [g[s]["ndcg10_normal"] for s in srcs], colors, "Greedy rewrites, 2,305 dev queries (nDCG@10, standard BM25)", "deck_headline.png", (7.4, 5.3), ylim=(0, 0.45))
    q = {r["source"]: r for r in S["greedy_eval"]["query_stats"]}
    bars(labels, [q[s]["any_repeat_rate"] for s in srcs], colors, "Greedy rewrites containing a repeated term", "deck_repeat.png", (7.4, 5.3), pct=True, ylim=(0, 0.5))

    m = S["means"]
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ks = [1, 2, 4, 8, 16]
    for src, color, label in (("untrained", HEX["untrained"], "untrained"), ("trained-normal", HEX["normal"], "normal"), ("trained-count_once", HEX["once"], "count-once")):
        ax.plot(ks, [m[src]["ndcg_normal_mean"]] + [m[src][f"best_of_{k}"] for k in ks[1:]], color=color, lw=2.2, marker="o", ms=6, label=label)
    ax.axhline(m["original"]["ndcg_normal_mean"], color=HEX["raw"], ls="--", lw=1.5, label="raw query")
    ax.set(xscale="log", title="Expected best-of-k nDCG@10 (32 samples per query)", xlabel="k samples", ylabel="nDCG@10")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    grid(ax)
    ax.legend(frameon=False)
    save(fig, "deck_best_of_k.png")

    rc = S["rewrite_class"]
    cls = ["compression", "same", "substitution", "mixed", "expansion"]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for i, (src, color, label) in enumerate((("untrained", HEX["untrained"], "untrained"), ("trained-normal", HEX["normal"], "normal"), ("trained-count_once", HEX["once"], "count-once"))):
        ax.bar([j + (i - 1) * 0.27 for j in range(len(cls))], [rc[src][c] for c in cls], 0.27, color=color, label=label)
    ax.set_xticks(range(len(cls)), cls)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Change in the query's terms (share of sampled rewrites)")
    grid(ax)
    ax.legend(frameon=False)
    save(fig, "deck_rewrite_class.png")
    print("wrote deck figures")


if __name__ == "__main__":
    main()
