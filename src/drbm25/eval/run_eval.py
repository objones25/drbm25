"""Score queries on the full dev set under both BM25 modes and write one JSON-lines file.

    python -m drbm25.eval.run_eval --config PATH --source {original,untrained,checkpoint}
        [--checkpoint STEP_DIR --label {trained-normal,trained-count_once}] --out FILE.jsonl
        [--limit N] [--set key=value ...]

``original`` scores the dataset queries as they are (CPU, no model). ``untrained`` rewrites them
with the base model (LoRA adapters at initialisation equal the base). ``checkpoint`` loads a
``step_NNNNNN`` checkpoint directory first: LoRA adapters (single-file layout) or, with a
``model.train_mode: full`` config, the actor of a sharded full fine-tune checkpoint (DCP files
read in this one process); the record source is ``trained-<reward_mode>``, read from the checkpoint. Rewrites are greedy, with ``rollout.max_new_tokens`` from the config.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final, get_args

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from drbm25.bm25.searcher import Bm25Searcher, SearchResult
from drbm25.checkpoint import checkpoint_layout, load_checkpoint, load_sharded_actor
from drbm25.config import Config, ScoringMode, load_config
from drbm25.data import DEV_FILE, Example, load_examples
from drbm25.eval.records import (
    SOURCE_ORIGINAL,
    SOURCE_UNTRAINED,
    QueryRecord,
    write_records,
)
from drbm25.obs import configure
from drbm25.ppo.models import LoraPolicyModels
from drbm25.ppo.trainer import Searcher, score_responses, search_all
from drbm25.prompt import build_prompt
from drbm25.reward import ParsedRewrite, ndcg_at_k
from drbm25.rollout.base import RolloutEngine
from drbm25.rollout.hf_generate import HFGenerateEngine
from drbm25.tiny import build_tiny_qwen2
from drbm25.train import (
    TINY_MODEL,
    build_base,
    build_tokenizer,
    check_index,
    check_java,
    check_layout,
    check_tokenizer,
    precision_dtype,
)

logger = logging.getLogger(__name__)

_MODES: Final[tuple[ScoringMode, ...]] = get_args(ScoringMode)
_SOURCE_CHECKPOINT: Final = "checkpoint"


def _search_error(results: dict[ScoringMode, SearchResult | None]) -> str | None:
    overflowed = [mode for mode in _MODES if results[mode] is None]
    return f"too_many_clauses:{','.join(overflowed)}" if overflowed else None


def _ndcg(result: SearchResult | None, example: Example, k: int) -> float:
    return 0.0 if result is None else ndcg_at_k(result.doc_ids, example.relevant_ids, k)


def score_original(examples: Sequence[Example], searcher: Searcher, *, k: int) -> list[QueryRecord]:
    """Score each example's own query text (nDCG@k) under both modes."""
    if not examples:
        raise ValueError("no examples to score")
    queries = [e.query for e in examples]
    found = {mode: search_all(searcher, queries, mode=mode, k=k) for mode in _MODES}
    records = []
    for i, example in enumerate(examples):
        row: dict[ScoringMode, SearchResult | None] = {mode: found[mode][i] for mode in _MODES}
        records.append(
            QueryRecord(
                qid=example.qid,
                source=SOURCE_ORIGINAL,
                query_text=example.query,
                rewrite_text=None,
                format_error=None,
                terms=tuple(searcher.analyze(example.query)),
                ndcg_normal=_ndcg(row["normal"], example, k),
                ndcg_count_once=_ndcg(row["count_once"], example, k),
                search_error=_search_error(row),
            )
        )
    return records


def score_rewrites(
    examples: Sequence[Example], engine: RolloutEngine, searcher: Searcher, *, k: int, source: str
) -> list[QueryRecord]:
    """Greedy rewrites scored under both modes; a format error scores 0.0 in both."""
    if not examples:
        raise ValueError("no examples to score")
    batch = engine.generate([build_prompt(e.query) for e in examples], greedy=True)
    if len(batch.response_text) != len(examples):
        raise ValueError(f"engine returned {len(batch.response_text)} responses for {len(examples)} examples")
    scored = score_responses(searcher, batch.response_text, examples, k=k)
    records = []
    for i, (example, parsed) in enumerate(zip(examples, scored.parsed, strict=True)):
        row: dict[ScoringMode, SearchResult | None] = {mode: scored.results[mode][i] for mode in _MODES}
        if isinstance(parsed, ParsedRewrite):
            rewrite_text, format_error = parsed.query, None
            terms, error = tuple(searcher.analyze(parsed.query)), _search_error(row)
        else:
            rewrite_text, format_error, terms, error = batch.response_text[i], parsed.reason.value, (), None
        records.append(
            QueryRecord(
                qid=example.qid,
                source=source,
                query_text=example.query,
                rewrite_text=rewrite_text,
                format_error=format_error,
                terms=terms,
                ndcg_normal=scored.ndcg["normal"][i],
                ndcg_count_once=scored.ndcg["count_once"][i],
                search_error=error,
            )
        )
    return records


# ---------------------------------------------------------------------------- CLI
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drbm25.eval.run_eval", description="Score dev queries under both BM25 modes."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", choices=(SOURCE_ORIGINAL, SOURCE_UNTRAINED, _SOURCE_CHECKPOINT), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="step_NNNNNN checkpoint directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N dev queries")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ns = parser.parse_args(argv)
    is_checkpoint = ns.source == _SOURCE_CHECKPOINT
    if is_checkpoint and ns.checkpoint is None:
        parser.error("--source checkpoint needs --checkpoint")
    if not is_checkpoint and ns.checkpoint is not None:
        parser.error("--checkpoint only goes with --source checkpoint")
    if ns.limit is not None and ns.limit < 1:
        parser.error(f"--limit must be >= 1, got {ns.limit}")
    overrides: dict[str, str] = {}
    for item in ns.overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--set expects key=value, got {item!r}")
        overrides[key] = value
    ns.overrides = overrides
    return ns


def trained_source(reward_mode: ScoringMode) -> str:
    return f"trained-{reward_mode}"


class _FrozenActor:
    """A plain HF causal LM as a ``GenerationSource`` (evaluation of a full fine-tune actor)."""

    def __init__(self, model: PreTrainedModel) -> None:
        self._model = model

    def generation_module(self) -> nn.Module:
        self._model.eval()
        return self._model


def _plain_actor(cfg: Config) -> PreTrainedModel:
    """The untrained actor, fp32 on CPU (a checkpoint is loaded into it before the cast)."""
    if cfg.model.name == TINY_MODEL:
        return build_tiny_qwen2(seed=cfg.run.seed)
    return AutoModelForCausalLM.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, dtype=torch.float32, attn_implementation="sdpa"
    )


def _build_full_engine(cfg: Config, checkpoint: Path | None, device: torch.device) -> tuple[HFGenerateEngine, str]:
    tokenizer = build_tokenizer(cfg)
    check_tokenizer(tokenizer)
    actor = _plain_actor(cfg)
    source = SOURCE_UNTRAINED
    if checkpoint is not None:
        state = load_sharded_actor(checkpoint, actor)
        source = trained_source(state.reward_mode)
        logger.info("loaded full fine-tune actor from %s (step %d, %s)", checkpoint, state.step, source)
    nn.Module.to(actor, device=device, dtype=precision_dtype(cfg))  # transformers' decorated .to() confuses pyright
    return HFGenerateEngine(_FrozenActor(actor), tokenizer, cfg.rollout, device), source


def _build_engine(cfg: Config, checkpoint: Path | None) -> tuple[HFGenerateEngine, str]:
    """Greedy rewriter for ``cfg``'s layout; a checkpoint must be of the matching layout.

    LoRA (``model.train_mode: lora``): base + adapters from a single-file checkpoint. Full fine-tune
    (``full``): the actor from a sharded checkpoint, loaded in this one process and run with HF
    ``generate`` in ``model.precision``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.run.seed)
    layout = None if checkpoint is None else checkpoint_layout(checkpoint)
    if cfg.model.train_mode == "full":
        if layout == "single":
            raise ValueError(f"{checkpoint} is not a sharded (full fine-tune) checkpoint; use the LoRA config")
        return _build_full_engine(cfg, checkpoint, device)
    if layout == "sharded":
        raise ValueError(f"{checkpoint} is a sharded full fine-tune checkpoint: use a config with train_mode 'full'")
    lora = check_layout(cfg, device)
    base, tokenizer = build_base(cfg, device)
    check_tokenizer(tokenizer)
    models = LoraPolicyModels(base, lora, gradient_checkpointing=False)
    source = SOURCE_UNTRAINED
    if checkpoint is not None:
        loaded = load_checkpoint(checkpoint)
        models.load_trainable_state_dict(loaded.model)
        source = trained_source(loaded.train_state.reward_mode)
        logger.info("loaded checkpoint %s (step %d, %s)", checkpoint, loaded.train_state.step, source)
    return HFGenerateEngine(models, tokenizer, cfg.rollout, device), source


def main(argv: Sequence[str] | None = None) -> int:
    ns = _parse_args(argv)
    configure(level="INFO", fmt="json")
    cfg = load_config(ns.config, ns.overrides)
    check_java()
    checksum = check_index(cfg)
    r = cfg.retrieval
    searcher = Bm25Searcher(r.index_dir, k1=r.k1, b=r.b, threads=r.search_threads, timeout_s=r.search_timeout_s)
    examples = load_examples(cfg.data.data_dir / DEV_FILE)[: ns.limit]
    logger.info("scoring %d dev queries, source=%s, index checksum %s", len(examples), ns.source, checksum)
    if ns.source == SOURCE_ORIGINAL:
        records = score_original(examples, searcher, k=r.eval_k)
    else:
        engine, source = _build_engine(cfg, ns.checkpoint)
        records = score_rewrites(examples, engine, searcher, k=r.eval_k, source=source)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    write_records(ns.out, records)
    logger.info("wrote %d records to %s", len(records), ns.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
