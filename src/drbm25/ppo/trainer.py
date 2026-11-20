"""One PPO step for the query rewriter: rollout -> BM25 reward -> GAE -> clipped updates.

``PPOTrainer.step`` runs on every rank with that rank's share of the global batch:

1. generate with the actor (sampling), parse each response, search BM25 in the reward mode
   (reward) and in the other mode (metric only), compute the scalar rewards;
2. concatenate prompt + response ids; old log-probs (actor), reference log-probs and values,
   all without grad;
3. KL-in-reward, GAE (gamma = lambda = 1), whiten the advantages;
4. per mini-batch: micro-batches with gradient accumulation, the actor loss
   ``pg - entropy_coef * entropy`` backpropagated *before* the critic forward (the LoRA layout
   with gradient checkpointing forbids a forward through the other adapter while a backward
   is pending), all-reduce, unscale, clip, step;
5. ``engine.sync_weights``.

Alignment (prompt width ``P``, response width ``R``, sequence ``L = P + R``): logits at position
``t`` predict token ``t + 1``, so response token ``j`` (absolute position ``P + j``) is scored by
the logits at ``P - 1 + j`` -> ``logits[:, P-1 : L-1]``. ``values[:, t]`` is the value of the
state after token ``t``; token ``j`` is chosen in the state after token ``P - 1 + j`` -> the
response values are ``values[:, P-1 : L-1]``. Both line up with the (unshifted) response mask.
All log-probs (old, reference, new) and the entropy are computed from ``logits / T`` with
``T = rollout.temperature``, the policy the rollouts were sampled from (as verl does).

Whitening is per rank: with 2 ranks each whitens its own 32 rollouts. That differs from verl's
whole-batch whitening by the spread of the two halves' statistics; it is accepted here because
the two halves are i.i.d. samples of the same data order.

Full fine-tune layout (``model.train_mode: full``, ``FullPolicyModels``): FSDP2 reduce-scatters
and averages the gradients inside ``backward``, so the manual all-reduce is skipped. The update
ratio needs a copy of the weights from before the step, which at 3B fp32 is 6 GB per rank per
model; on this layout it is computed on deep-log steps only, for the actor only (the critic's
key is omitted), from a host-memory copy of this rank's shards. Both layouts drop the gradients
(``zero_grad(set_to_none=True)``) after the update, before ``sync_weights`` gathers the actor.

Metrics are reduced over ranks (mean; min/max for ``*/min``, ``*/max``, ``ppo/kl_ref_max``
and ``sys/*``), so the health guards decide the same way on every rank.
"""

from __future__ import annotations

import logging
import math
import resource
import sys
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Final, Protocol, get_args

import torch
import torch.distributed as tdist
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler

from drbm25.bm25.searcher import SearchResult, TooManyClausesError
from drbm25.config import Config, ScoringMode
from drbm25.data import Example
from drbm25.distributed import DistInfo, allreduce_mean_grads
from drbm25.obs.metrics import (
    ActivationProbe,
    explained_variance,
    per_layer_grad_and_weight_norms,
    qk_logit_bound,
    sharded_update_ratio,
    snapshot,
    snapshot_local_shards_to_cpu,
    update_ratio,
)
from drbm25.ppo.gae import gae, kl_penalized_rewards
from drbm25.ppo.losses import entropy_from_logits, masked_mean, masked_whiten, policy_loss, value_loss
from drbm25.ppo.models import PolicyModels
from drbm25.prompt import build_prompt
from drbm25.reward import (
    FORMAT_BONUS,
    REWARD_FORMAT_ERROR,
    FormatError,
    FormatErrorReason,
    ParsedRewrite,
    compute_reward,
    ndcg_at_k,
    parse_response,
)
from drbm25.rollout.base import RolloutBatch, RolloutEngine

logger = logging.getLogger(__name__)

REWARD_MIN: Final = REWARD_FORMAT_ERROR
REWARD_MAX: Final = FORMAT_BONUS + 1.0
HEALTH_STREAK: Final = 5  # consecutive unhealthy steps before TrainingHealthError
SCALE_FLOOR: Final = 1.0
APPROX_KL_MAX: Final = 1.0
N_VAL_SAMPLES: Final = 8
_MODES: Final[tuple[ScoringMode, ...]] = get_args(ScoringMode)
_GB: Final = 1e9


class NonFiniteError(RuntimeError):
    """NaN/Inf in a loss, log-probs, values or advantages; the message names the tensor and step."""


class TrainingHealthError(RuntimeError):
    """GradScaler scale or approx KL stayed out of bounds for ``HEALTH_STREAK`` consecutive steps."""


class Searcher(Protocol):
    """The part of ``Bm25Searcher`` the trainer uses (tests pass fakes)."""

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult: ...

    def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]: ...

    def analyze(self, text: str) -> list[str]: ...


@dataclass(frozen=True)
class ValSample:
    """One row of the W&B ``samples`` table."""

    query: str
    rewrite: str
    analyzed_terms: str
    reward_normal: float
    reward_count_once: float


@dataclass(frozen=True)
class ScoredBatch:
    """Per-row scoring of one batch of responses under both BM25 modes."""

    parsed: list[ParsedRewrite | FormatError]
    results: dict[ScoringMode, list[SearchResult | None]]  # None: parse failed or too many clauses
    ndcg: dict[ScoringMode, list[float]]  # 0.0 for format errors and overflowing queries
    rewards: dict[ScoringMode, list[float]]
    search_s: float


def search_all(searcher: Searcher, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult | None]:
    """Search every query; ``None`` marks a query over Lucene's clause limit (reward 0, counted)."""
    if not queries:
        return []
    try:
        return list(searcher.batch_search(queries, mode=mode, k=k))
    except TooManyClausesError:
        # batch_search surfaces only the first failure; redo one by one to find every overflow.
        logger.warning("a %s query exceeded the clause limit; re-searching one by one", mode, exc_info=True)
    out: list[SearchResult | None] = []
    for query in queries:
        try:
            out.append(searcher.search(query, mode=mode, k=k))
        except TooManyClausesError:
            logger.warning("query over the clause limit scored 0 (%s mode)", mode, exc_info=True)
            out.append(None)
    return out


def score_responses(
    searcher: Searcher, texts: Sequence[str], examples: Sequence[Example], *, k: int
) -> ScoredBatch:
    """Parse each response, search the rewrites under both modes, and score nDCG@k and reward."""
    parsed = [parse_response(t) for t in texts]
    queries = [(i, p.query) for i, p in enumerate(parsed) if isinstance(p, ParsedRewrite)]
    results: dict[ScoringMode, list[SearchResult | None]] = {}
    start = time.perf_counter()
    for mode in _MODES:
        found = search_all(searcher, [q for _, q in queries], mode=mode, k=k)
        per_row: list[SearchResult | None] = [None] * len(parsed)
        for (i, _), result in zip(queries, found, strict=True):
            per_row[i] = result
        results[mode] = per_row
    search_s = time.perf_counter() - start
    ndcg: dict[ScoringMode, list[float]] = {}
    rewards: dict[ScoringMode, list[float]] = {}
    for mode in _MODES:
        ndcg[mode] = [
            0.0 if r is None else ndcg_at_k(r.doc_ids, e.relevant_ids, k)
            for r, e in zip(results[mode], examples, strict=True)
        ]
        rewards[mode] = [
            compute_reward(p, None if isinstance(p, FormatError) else n)
            for p, n in zip(parsed, ndcg[mode], strict=True)
        ]
    return ScoredBatch(parsed=parsed, results=results, ndcg=ndcg, rewards=rewards, search_s=search_s)


def tempered_response_logits(logits: Tensor, *, prompt_len: int, temperature: float) -> Tensor:
    """``[B, R, V]`` fp32 ``logits / temperature`` at the positions that predict response tokens.

    Rollouts sample from ``softmax(logits / T)``, so old, reference and new log-probs and the
    entropy must all score that tempered policy (verl divides by ``rollout.temperature`` in
    ``dp_actor.py``). Every log-prob and entropy in the trainer goes through this one helper.
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    _require_response(logits.shape[1], prompt_len)
    return logits[:, prompt_len - 1 : -1].float() / temperature


def response_logprobs(logits: Tensor, input_ids: Tensor, *, prompt_len: int, temperature: float) -> Tensor:
    """``[B, R]`` fp32 log-prob of each response token under ``softmax(logits / temperature)``."""
    return _gather_logprobs(
        tempered_response_logits(logits, prompt_len=prompt_len, temperature=temperature), input_ids, prompt_len
    )


def _gather_logprobs(tempered: Tensor, input_ids: Tensor, prompt_len: int) -> Tensor:
    response = input_ids[:, prompt_len:]
    if tempered.shape[:2] != response.shape:
        raise ValueError(f"tempered logits {tuple(tempered.shape)} do not match response ids {tuple(response.shape)}")
    return torch.log_softmax(tempered, dim=-1).gather(-1, response.unsqueeze(-1)).squeeze(-1)


def response_values(values: Tensor, *, prompt_len: int) -> Tensor:
    """``[B, R]`` value of the state in which each response token is chosen: ``values[:, P-1 : L-1]``."""
    _require_response(values.shape[1], prompt_len)
    return values[:, prompt_len - 1 : -1]


def _require_response(length: int, prompt_len: int) -> None:
    if prompt_len < 1 or length <= prompt_len:
        raise ValueError(f"sequence of length {length} has no response tokens after prompt_len={prompt_len}")


def _require_finite(name: str, value: Tensor, step: int) -> None:
    if not bool(torch.isfinite(value).all()):
        raise NonFiniteError(f"non-finite {name} at step {step}")


def _mean(values: Sequence[float]) -> float:
    """Mean, or NaN when there is nothing to average (an honest gap in the W&B series)."""
    return sum(values) / len(values) if values else math.nan


def _host_rss_peak_gb() -> float:
    # ru_maxrss is bytes on macOS and KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / _GB if sys.platform == "darwin" else peak * 1024 / _GB


class PPOTrainer:
    """PPO over ``models`` (actor + critic + reference) with a BM25 nDCG@10 reward.

    ``step`` expects this rank's ``batch_size // world_size`` examples. Optimizers are AdamW
    with no weight decay and a constant learning rate per group (the spec sets none).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        models: PolicyModels,
        engine: RolloutEngine,
        searcher: Searcher,
        reward_mode: ScoringMode,
        dist: DistInfo,
        scaler: GradScaler | None,
    ) -> None:
        if reward_mode not in _MODES:
            raise ValueError(f"reward_mode must be one of {_MODES}, got {reward_mode!r}")
        self._cfg = cfg
        self._models = models
        self._engine = engine
        self._searcher = searcher
        self._reward_mode: ScoringMode = reward_mode
        self._dist = dist
        self._scaler = scaler
        self._actor_params = models.actor_parameters()
        self._critic_params = models.critic_parameters()
        self._actor_opt = torch.optim.AdamW(self._actor_params, lr=cfg.ppo.actor_lr, weight_decay=0.0)
        self._critic_opt = torch.optim.AdamW(self._critic_params, lr=cfg.ppo.critic_lr, weight_decay=0.0)
        self._local_batch = cfg.ppo.batch_size // dist.world_size
        self._local_mini = cfg.ppo.mini_batch_size // dist.world_size
        self._micro = cfg.ppo.micro_batch_size
        if self._local_batch % self._local_mini or self._local_mini % self._micro:
            raise ValueError(
                f"per-rank batch {self._local_batch}, mini-batch {self._local_mini} and micro-batch "
                f"{self._micro} must divide each other; run validate_for_runtime first"
            )
        self._temperature = cfg.rollout.temperature  # training scores the sampling policy pi_T
        self._fsdp = cfg.model.train_mode == "full"  # FSDP2 averages grads in backward
        self._low_scale_streak = 0
        self._high_kl_streak = 0
        self._histograms: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ public API
    def step(self, examples: Sequence[Example], *, step: int) -> dict[str, float]:
        """Run one PPO step on this rank's examples; return metrics reduced over ranks."""
        if len(examples) != self._local_batch:
            raise ValueError(f"expected {self._local_batch} examples for this rank, got {len(examples)}")
        timings: dict[str, float] = {}
        if self._dist.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._dist.device)

        with self._timed(timings, "generate"):
            rollout = self._engine.generate([build_prompt(e.query) for e in examples], greedy=False)
        scored = self._score(rollout.response_text, examples)
        timings["search"] = scored.search_s
        rewards = self._checked_rewards(scored.rewards[self._reward_mode], step)

        prompt_len = rollout.prompt_ids.shape[1]
        ids = torch.cat([rollout.prompt_ids, rollout.response_ids], dim=1)
        attn = torch.cat([rollout.prompt_mask, rollout.response_mask], dim=1)
        mask = rollout.response_mask.to(torch.float32)

        with self._timed(timings, "logprobs"):
            old = self._no_grad_logprobs(self._models.actor_logits, ids, attn, prompt_len)
            _require_finite("logprobs_old", old, step)
            ref = self._no_grad_logprobs(self._models.ref_logits, ids, attn, prompt_len)
            _require_finite("logprobs_ref", ref, step)
        with self._timed(timings, "values"):
            values = self._no_grad_values(ids, attn, prompt_len)
            _require_finite("values_old", values, step)

        token_rewards = kl_penalized_rewards(rewards, old, ref, mask, kl_coef=self._cfg.ppo.kl_coef)
        adv_raw, returns = gae(token_rewards, values, mask, gamma=self._cfg.ppo.gamma, lam=self._cfg.ppo.lam)
        _require_finite("advantages", adv_raw, step)
        advantages = masked_whiten(adv_raw, mask) * mask
        _require_finite("advantages", advantages, step)

        deep = step % self._cfg.run.deep_log_every == 0
        with self._timed(timings, "update"):
            update = self._update(ids, attn, mask, old, values, advantages, returns, prompt_len, step=step, deep=deep)
        with self._timed(timings, "sync"):
            self._engine.sync_weights(self._models)

        metrics: dict[str, float] = {}
        metrics.update(self._reward_metrics(scored, rewards))
        metrics.update(self._ppo_metrics(old, ref, mask, adv_raw, returns, values))
        metrics.update(update)
        metrics.update(self._rollout_metrics(rollout, timings["generate"]))
        metrics.update({f"time/{name}": seconds for name, seconds in timings.items()})
        metrics.update(self._sys_metrics())
        self._histograms = {}
        if deep:
            metrics.update(self._deep_metrics(ids[: self._micro], attn[: self._micro]))
            valid = mask.bool()
            self._histograms = {
                "hist/kl_ref": (old - ref)[valid].tolist(),
                "hist/advantages": adv_raw[valid].tolist(),
                "hist/response_len": mask.sum(dim=1).tolist(),
            }
        metrics = self._reduce(metrics)
        self._check_health(metrics, step)
        return metrics

    def histograms(self) -> dict[str, list[float]]:
        """This rank's per-token KL to ref, raw advantages (valid tokens) and response lengths.

        Filled on deep-log steps only (empty otherwise); rank-local, not reduced. For W&B histograms.
        """
        return {key: list(values) for key, values in self._histograms.items()}

    def validate_with_samples(self, examples: Sequence[Example]) -> tuple[dict[str, float], list[ValSample]]:
        """Greedy rollouts scored with nDCG@10 under both modes, plus the first ``N_VAL_SAMPLES`` rows."""
        if not examples:
            raise ValueError("validate needs at least one example")
        rollout = self._engine.generate([build_prompt(e.query) for e in examples], greedy=True)
        scored = self._score(rollout.response_text, examples, k=self._cfg.retrieval.eval_k)
        metrics = {f"val/ndcg_{mode}": _mean(scored.ndcg[mode]) for mode in _MODES}
        samples = []
        for i, example in enumerate(examples[:N_VAL_SAMPLES]):
            parsed = scored.parsed[i]
            terms = self._searcher.analyze(parsed.query) if isinstance(parsed, ParsedRewrite) else []
            samples.append(
                ValSample(
                    query=example.query,
                    rewrite=rollout.response_text[i],
                    analyzed_terms=" ".join(terms),
                    reward_normal=scored.rewards["normal"][i],
                    reward_count_once=scored.rewards["count_once"][i],
                )
            )
        return metrics, samples

    def validate(self, examples: Sequence[Example]) -> dict[str, float]:
        return self.validate_with_samples(examples)[0]

    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """The live optimizers, keyed ``"actor"`` / ``"critic"`` (for sharded checkpoints)."""
        return {"actor": self._actor_opt, "critic": self._critic_opt}

    def optimizer_states(self) -> dict[str, dict[str, Any]]:
        return {"actor": self._actor_opt.state_dict(), "critic": self._critic_opt.state_dict()}

    def load_optimizer_states(self, states: Mapping[str, dict[str, Any]]) -> None:
        missing = sorted({"actor", "critic"} - states.keys())
        if missing:
            raise KeyError(f"optimizer states missing {missing}")
        self._actor_opt.load_state_dict(states["actor"])
        self._critic_opt.load_state_dict(states["critic"])

    # ------------------------------------------------------------------ scoring
    def _score(self, texts: Sequence[str], examples: Sequence[Example], *, k: int | None = None) -> ScoredBatch:
        k = self._cfg.retrieval.train_k if k is None else k
        return score_responses(self._searcher, texts, examples, k=k)

    def _checked_rewards(self, rewards: Sequence[float], step: int) -> Tensor:
        tensor = torch.tensor(rewards, dtype=torch.float32, device=self._dist.device)
        bad = ~torch.isfinite(tensor) | (tensor < REWARD_MIN) | (tensor > REWARD_MAX)
        if bool(bad.any()):
            value = float(tensor[bad][0])
            raise AssertionError(f"reward {value} outside [{REWARD_MIN}, {REWARD_MAX}] at step {step}")
        return tensor

    # ------------------------------------------------------------------ forwards
    def _autocast(self) -> AbstractContextManager[object]:
        precision = self._cfg.model.precision
        if precision == "fp32" or self._dist.device.type != "cuda":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _chunks(self, n: int) -> list[slice]:
        return [slice(i, i + self._micro) for i in range(0, n, self._micro)]

    @torch.no_grad()
    def _no_grad_logprobs(
        self, forward: Callable[[Tensor, Tensor], Tensor], ids: Tensor, attn: Tensor, prompt_len: int
    ) -> Tensor:
        parts = []
        for chunk in self._chunks(ids.shape[0]):
            with self._autocast():
                logits = forward(ids[chunk], attn[chunk])
            parts.append(response_logprobs(logits, ids[chunk], prompt_len=prompt_len, temperature=self._temperature))
        return torch.cat(parts)

    @torch.no_grad()
    def _no_grad_values(self, ids: Tensor, attn: Tensor, prompt_len: int) -> Tensor:
        parts = []
        for chunk in self._chunks(ids.shape[0]):
            with self._autocast():
                values = self._models.critic_values(ids[chunk], attn[chunk])
            parts.append(response_values(values.float(), prompt_len=prompt_len))
        return torch.cat(parts)

    # ------------------------------------------------------------------ update
    def _backward(self, loss: Tensor) -> None:
        if self._scaler is None:
            loss.backward()
        else:
            self._scaler.scale(loss).backward()

    def _update(
        self,
        ids: Tensor,
        attn: Tensor,
        mask: Tensor,
        old: Tensor,
        values: Tensor,
        advantages: Tensor,
        returns: Tensor,
        prompt_len: int,
        *,
        step: int,
        deep: bool,
    ) -> dict[str, float]:
        ppo = self._cfg.ppo
        n_micro = self._local_mini // self._micro
        stats: dict[str, list[float]] = {}
        ratios = self._update_ratio_snapshots(deep=deep)
        layer_norms: dict[str, float] = {}
        for _epoch in range(ppo.ppo_epochs):
            for start in range(0, ids.shape[0], self._local_mini):
                self._actor_opt.zero_grad(set_to_none=True)
                self._critic_opt.zero_grad(set_to_none=True)
                for chunk in self._chunks(self._local_mini):
                    rows = slice(start + chunk.start, start + chunk.stop)
                    self._micro_step(
                        ids[rows], attn[rows], mask[rows], old[rows], values[rows], advantages[rows],
                        returns[rows], prompt_len, n_micro=n_micro, step=step, stats=stats,
                    )  # fmt: skip
                optim_stats, norms = self._optimizer_step(step, deep=deep)
                for key, value in optim_stats.items():
                    stats.setdefault(key, []).append(value)
                layer_norms = norms or layer_norms  # deep steps: the last mini-batch's norms
        # The grads are not read after the step; dropping them frees memory before sync_weights.
        self._actor_opt.zero_grad(set_to_none=True)
        self._critic_opt.zero_grad(set_to_none=True)
        out = {key: _mean(vals) for key, vals in stats.items()}
        out.update(ratios())
        out["optim/actor_lr"] = float(self._actor_opt.param_groups[0]["lr"])
        out["optim/critic_lr"] = float(self._critic_opt.param_groups[0]["lr"])
        out.update(layer_norms)
        return out

    def _update_ratio_snapshots(self, *, deep: bool) -> Callable[[], dict[str, float]]:
        """Snapshot the weights now; the returned callable gives the update-ratio metrics after the step."""
        if not self._fsdp:
            actor_before, critic_before = snapshot(self._actor_params), snapshot(self._critic_params)
            return lambda: {
                "optim/actor_update_ratio": update_ratio(actor_before, self._actor_params),
                "optim/critic_update_ratio": update_ratio(critic_before, self._critic_params),
            }
        if not deep:
            return dict
        actor_cpu = snapshot_local_shards_to_cpu(self._actor_params)
        return lambda: {
            "optim/actor_update_ratio": sharded_update_ratio(actor_cpu, self._actor_params, self._dist),
        }

    def _micro_step(
        self,
        ids: Tensor,
        attn: Tensor,
        mask: Tensor,
        old: Tensor,
        values_old: Tensor,
        advantages: Tensor,
        returns: Tensor,
        prompt_len: int,
        *,
        n_micro: int,
        step: int,
        stats: dict[str, list[float]],
    ) -> None:
        ppo = self._cfg.ppo
        # Actor forward and backward first; the critic forward may only start after it (see module doc).
        with self._autocast():
            logits = self._models.actor_logits(ids, attn)
        tempered = tempered_response_logits(logits, prompt_len=prompt_len, temperature=self._temperature)
        del logits
        new = _gather_logprobs(tempered, ids, prompt_len)
        _require_finite("logprobs_new", new, step)
        entropy = entropy_from_logits(tempered, mask)
        pg_loss, pg_stats = policy_loss(new, old, advantages, mask, clip_ratio=ppo.clip_ratio)
        actor_loss = pg_loss - ppo.entropy_coef * entropy
        _require_finite("policy_loss", actor_loss, step)
        self._backward(actor_loss / n_micro)
        del tempered

        with self._autocast():
            values_all = self._models.critic_values(ids, attn)
        values_new = response_values(values_all.float(), prompt_len=prompt_len)
        _require_finite("values_new", values_new, step)
        v_loss, v_stats = value_loss(values_new, values_old, returns, mask, clip_range=ppo.value_clip)
        _require_finite("value_loss", v_loss, step)
        self._backward(v_loss / n_micro)

        for key, value in (
            ("ppo/policy_loss", pg_loss),
            ("ppo/value_loss", v_loss),
            ("ppo/entropy", entropy),
            ("ppo/approx_kl", pg_stats["approx_kl"]),
            ("ppo/clipfrac", pg_stats["clipfrac"]),
            ("ppo/vf_clipfrac", v_stats["vf_clipfrac"]),
        ):
            stats.setdefault(key, []).append(float(value.detach()))

    def _optimizer_step(self, step: int, *, deep: bool) -> tuple[dict[str, float], dict[str, float]]:
        """All-reduce, unscale, clip, step. On deep-log steps also return per-layer norms.

        The per-layer grad norms are taken after unscale and BEFORE clipping, so they add up
        to the logged (pre-clip) ``optim/*_grad_norm``.
        """
        if not self._fsdp:
            allreduce_mean_grads(self._actor_params, self._dist)
            allreduce_mean_grads(self._critic_params, self._dist)
        clip = self._cfg.ppo.grad_clip
        if self._scaler is not None:
            self._scaler.unscale_(self._actor_opt)
            self._scaler.unscale_(self._critic_opt)
        layer_norms: dict[str, float] = {}
        if deep:
            layers = self._models.decoder_layers()
            layer_norms = {
                **per_layer_grad_and_weight_norms(layers, "layers/actor", only=self._actor_params),
                **per_layer_grad_and_weight_norms(layers, "layers/critic", only=self._critic_params),
            }
        actor_norm = float(nn.utils.clip_grad_norm_(self._actor_params, clip))
        critic_norm = float(nn.utils.clip_grad_norm_(self._critic_params, clip))
        skipped = 0.0
        if self._scaler is None:
            # Without a scaler nothing skips a non-finite step, so it must stop the run here.
            _require_finite("actor_grad_norm", torch.tensor(actor_norm), step)
            _require_finite("critic_grad_norm", torch.tensor(critic_norm), step)
            self._actor_opt.step()
            self._critic_opt.step()
            scale = 1.0
        else:
            before = self._scaler.get_scale()
            self._scaler.step(self._actor_opt)
            self._scaler.step(self._critic_opt)
            self._scaler.update()
            scale = self._scaler.get_scale()
            skipped = float(scale < before)  # update() backs the scale off only after an inf/NaN grad
        stats = {
            "optim/actor_grad_norm": actor_norm,
            "optim/critic_grad_norm": critic_norm,
            "optim/actor_clipped": float(actor_norm > clip),
            "optim/critic_clipped": float(critic_norm > clip),
            "optim/scaler_scale": scale,
            "optim/scaler_skipped": skipped,
        }
        return stats, layer_norms

    # ------------------------------------------------------------------ metrics
    def _reward_metrics(self, scored: ScoredBatch, rewards: Tensor) -> dict[str, float]:
        n = len(scored.parsed)
        mode = self._reward_mode
        reasons = [p.reason for p in scored.parsed if isinstance(p, FormatError)]
        out = {
            "reward/mean": float(rewards.mean()),
            "reward/std": float(rewards.std(correction=0)),
            "reward/min": float(rewards.min()),
            "reward/max": float(rewards.max()),
            "reward/format_error_rate": len(reasons) / n,
            "reward/zero_ndcg_rate": sum(
                isinstance(p, ParsedRewrite) and v == 0.0 for p, v in zip(scored.parsed, scored.ndcg[mode], strict=True)
            )
            / n,
        }
        out.update({f"reward/format_error/{r.value}": reasons.count(r) / n for r in FormatErrorReason})
        out.update({f"ndcg/train_{m}": _mean(scored.ndcg[m]) for m in _MODES})
        # Analysis is identical in both modes; a query that overflowed in one mode did in both.
        searched = [r for r in scored.results[mode] if r is not None]
        nonempty = [r for r in searched if r.n_terms > 0]
        out["rewrite/n_terms"] = _mean([r.n_terms for r in searched])
        out["rewrite/n_unique_terms"] = _mean([r.n_unique_terms for r in searched])
        out["rewrite/repeat_ratio"] = _mean([1.0 - r.n_unique_terms / r.n_terms for r in nonempty])
        out["rewrite/any_repeat_rate"] = _mean([float(r.n_unique_terms < r.n_terms) for r in nonempty])
        n_parsed = sum(isinstance(p, ParsedRewrite) for p in scored.parsed)
        out["search/empty_query_rate"] = sum(r.n_terms == 0 for r in searched) / n
        out["search/too_many_clauses_rate"] = (n_parsed - len(searched)) / n
        out["search/seconds"] = scored.search_s
        return out

    def _ppo_metrics(
        self, old: Tensor, ref: Tensor, mask: Tensor, adv_raw: Tensor, returns: Tensor, values: Tensor
    ) -> dict[str, float]:
        kl = old - ref
        valid = mask.bool()
        adv_valid = adv_raw[valid]
        return {
            "ppo/kl_ref_mean": float(masked_mean(kl, mask)),
            "ppo/kl_ref_max": float(kl[valid].max()),
            "ppo/adv_mean_raw": float(adv_valid.mean()),
            "ppo/adv_std_raw": float(adv_valid.std(correction=0)),
            "ppo/returns_mean": float(masked_mean(returns, mask)),
            "ppo/values_mean": float(masked_mean(values, mask)),
            "ppo/explained_var": explained_variance(values, returns, mask),
        }

    @staticmethod
    def _rollout_metrics(rollout: RolloutBatch, generate_s: float) -> dict[str, float]:
        lengths = rollout.response_mask.sum(dim=1).float()
        return {
            "rollout/response_len_mean": float(lengths.mean()),
            "rollout/hit_max_tokens_rate": float(rollout.hit_max_tokens.float().mean()),
            "rollout/tokens_per_s": float(lengths.sum()) / generate_s if generate_s > 0 else math.nan,
        }

    def _sys_metrics(self) -> dict[str, float]:
        cuda = self._dist.device.type == "cuda"
        return {
            "sys/gpu_mem_peak_gb": torch.cuda.max_memory_allocated(self._dist.device) / _GB if cuda else 0.0,
            "sys/gpu_mem_reserved_gb": torch.cuda.memory_reserved(self._dist.device) / _GB if cuda else 0.0,
            "sys/host_rss_gb": _host_rss_peak_gb(),
        }

    @torch.no_grad()
    def _deep_metrics(self, ids: Tensor, attn: Tensor) -> dict[str, float]:
        """Activation stats, the QK-logit bound and the final-layer output norm on one actor micro-batch.

        The QK bound is computed inside a pre-hook so it sees the actor adapter (active during
        the actor forward). ``stab/final_norm_out`` is the mean per-token L2 norm of the last
        decoder layer's output, before the final RMSNorm (the norm whose growth precedes
        divergence; the post-norm output is normalised by construction).
        """
        layers = list(self._models.decoder_layers())
        bounds: list[float] = []
        final: list[float] = []

        def pre_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden = args[0] if args else kwargs["hidden_states"]
            bounds.append(qk_logit_bound(module, hidden))

        def out_hook(_module: nn.Module, _args: tuple[Any, ...], output: Tensor | tuple[Tensor, ...]) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            final.append(float(torch.linalg.vector_norm(hidden.float(), dim=-1).mean()))

        handles = [layer.register_forward_pre_hook(pre_hook, with_kwargs=True) for layer in layers]
        handles.append(layers[-1].register_forward_hook(out_hook))
        try:
            with ActivationProbe(layers, "act/actor") as probe, self._autocast():
                self._models.actor_logits(ids, attn)
            stats = probe.stats()
        finally:
            for handle in handles:
                handle.remove()
        stats["stab/qk_logit_bound_max"] = max(bounds)
        stats["stab/final_norm_out"] = final[0]
        return stats

    def _reduce(self, metrics: dict[str, float]) -> dict[str, float]:
        """Mean over ranks; min/max for the extreme-value keys. Every rank must pass the same keys."""
        if self._dist.world_size == 1:
            return metrics
        keys = sorted(metrics)
        n_keys = torch.tensor([len(keys), -len(keys)], dtype=torch.float64, device=self._dist.device)
        tdist.all_reduce(n_keys, op=tdist.ReduceOp.MAX)
        if int(n_keys[0]) != len(keys) or int(n_keys[1]) != -len(keys):
            raise RuntimeError(f"ranks disagree on the metric keys ({len(keys)} here)")

        def op_for(key: str) -> tdist.ReduceOp.RedOpType:
            if key.endswith(("/min",)):
                return tdist.ReduceOp.MIN
            if key.endswith(("/max", "/kl_ref_max")) or key.startswith("sys/"):
                return tdist.ReduceOp.MAX
            return tdist.ReduceOp.SUM

        out: dict[str, float] = {}
        for op in (tdist.ReduceOp.SUM, tdist.ReduceOp.MIN, tdist.ReduceOp.MAX):
            group = [k for k in keys if op_for(k) == op]
            values = torch.tensor([metrics[k] for k in group], dtype=torch.float64, device=self._dist.device)
            tdist.all_reduce(values, op=op)
            if op == tdist.ReduceOp.SUM:
                values /= self._dist.world_size
            out.update(zip(group, values.tolist(), strict=True))
        return out

    def _check_health(self, metrics: Mapping[str, float], step: int) -> None:
        scale = metrics["optim/scaler_scale"]
        approx_kl = metrics["ppo/approx_kl"]
        # Without a GradScaler the reported scale is 1.0, which is never below the floor.
        self._low_scale_streak = self._low_scale_streak + 1 if scale < SCALE_FLOOR else 0
        self._high_kl_streak = self._high_kl_streak + 1 if approx_kl > APPROX_KL_MAX else 0
        if self._low_scale_streak >= HEALTH_STREAK:
            raise TrainingHealthError(
                f"GradScaler scale {scale} < {SCALE_FLOOR} for {HEALTH_STREAK} consecutive steps (at step {step})"
            )
        if self._high_kl_streak >= HEALTH_STREAK:
            raise TrainingHealthError(
                f"approx_kl {approx_kl} > {APPROX_KL_MAX} for {HEALTH_STREAK} consecutive steps (at step {step})"
            )

    @contextmanager
    def _timed(self, timings: dict[str, float], name: str) -> Generator[None]:
        start = time.perf_counter()
        yield
        if self._dist.device.type == "cuda":
            torch.cuda.synchronize(self._dist.device)  # CUDA is async; time the work, not the launch
        timings[name] = time.perf_counter() - start
