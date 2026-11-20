"""PPOTrainer.step on the tiny Qwen2 + LoRA layout: metrics, guards, reward modes, learning."""

import dataclasses
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

import drbm25.ppo.trainer as trainer_module
from drbm25.bm25.searcher import Bm25Searcher, SearchResult, TooManyClausesError
from drbm25.config import Config, ScoringMode, load_config
from drbm25.data import Example
from drbm25.distributed import DistInfo
from drbm25.ppo.models import LoraPolicyModels, PolicyModels
from drbm25.ppo.trainer import (
    NonFiniteError,
    PPOTrainer,
    TrainingHealthError,
    response_logprobs,
    response_values,
)
from drbm25.rollout.base import RolloutBatch, RolloutEngine
from drbm25.rollout.hf_generate import HFGenerateEngine
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke_cpu.yaml"
CPU = torch.device("cpu")
DIST = DistInfo(rank=0, world_size=1, local_rank=0, device=CPU)

EVERY_STEP_KEYS = [
    *(f"reward/{s}" for s in ("mean", "std", "min", "max")),
    "reward/format_error_rate",
    "reward/zero_ndcg_rate",
    "ndcg/train_normal",
    "ndcg/train_count_once",
    *(f"rewrite/{s}" for s in ("n_terms", "n_unique_terms", "repeat_ratio", "any_repeat_rate")),
    *(f"search/{s}" for s in ("empty_query_rate", "too_many_clauses_rate", "seconds")),
    *(
        f"ppo/{s}"
        for s in (
            "policy_loss",
            "value_loss",
            "entropy",
            "kl_ref_mean",
            "kl_ref_max",
            "approx_kl",
            "clipfrac",
            "vf_clipfrac",
            "explained_var",
            "adv_mean_raw",
            "adv_std_raw",
            "returns_mean",
            "values_mean",
        )
    ),
    *(
        f"optim/{s}"
        for s in (
            "actor_lr",
            "critic_lr",
            "actor_grad_norm",
            "critic_grad_norm",
            "actor_clipped",
            "critic_clipped",
            "actor_update_ratio",
            "critic_update_ratio",
            "scaler_scale",
            "scaler_skipped",
        )
    ),
    *(f"rollout/{s}" for s in ("response_len_mean", "hit_max_tokens_rate", "tokens_per_s")),
    *(f"time/{s}" for s in ("generate", "logprobs", "values", "search", "update", "sync")),
    *(f"sys/{s}" for s in ("gpu_mem_peak_gb", "gpu_mem_reserved_gb", "host_rss_gb")),
]
FORMAT_ERROR_KEYS = [
    f"reward/format_error/{r}"
    for r in ("think_reopened", "tag_count", "tag_order", "bad_json", "no_query_key", "query_not_string")
]
DEEP_KEYS_PRESENT = ["stab/qk_logit_bound_max", "stab/final_norm_out", "layers/actor/layer_00/attn/weight_norm"]


# --------------------------------------------------------------------------- fakes and builders
class AnswerFormatEngine:
    """Real tiny-model generation, with each response text wrapped in a well-formed answer.

    The tiny model never writes the ``<answer>`` format, so without this every reward is -2.
    The query is the decoded response (``w*`` words, which also occur in the toy index).
    """

    def __init__(self, inner: RolloutEngine) -> None:
        self._inner = inner

    def generate(self, prompts: Sequence[str], *, greedy: bool) -> RolloutBatch:
        batch = self._inner.generate(prompts, greedy=greedy)
        texts = tuple(f'</think>\n<answer>{{"query": "{t}"}}</answer>' for t in batch.response_text)
        return dataclasses.replace(batch, response_text=texts)

    def sync_weights(self, models: PolicyModels) -> None:
        self._inner.sync_weights(models)


class ModeSearcher:
    """``normal`` ranks ``docN`` first, ``count_once`` ranks ``docC`` first, for every query."""

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult:
        del k
        doc = "docN" if mode == "normal" else "docC"
        return SearchResult((doc,), len(query.split()), len(set(query.split())), (1.0,))

    def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]:
        return [self.search(q, mode=mode, k=k) for q in queries]

    def analyze(self, text: str) -> list[str]:
        return text.split()


class TokenSevenSearcher(ModeSearcher):
    """The relevant doc ``hit`` is ranked first iff the query contains ``w4`` (tiny-tokenizer id 7)."""

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult:
        del mode, k
        words = query.split()
        doc = "hit" if "w4" in words else "miss"
        return SearchResult((doc,), len(words), len(set(words)), (1.0,))


class OverflowSearcher(ModeSearcher):
    """Every other query (1st, 3rd, ... of each mode) overflows Lucene's clause limit.

    ``batch_search`` therefore always raises, like the real one when any query overflows.
    """

    def __init__(self) -> None:
        self._calls: dict[str, int] = {"normal": 0, "count_once": 0}

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult:
        self._calls[mode] += 1
        if self._calls[mode] % 2 == 1:
            raise TooManyClausesError("too many clauses")
        return super().search(query, mode=mode, k=k)

    def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]:
        del mode, k
        raise TooManyClausesError(f"one of {len(queries)} queries has too many clauses")


def smoke_cfg(toy_index: Path, **overrides: str) -> Config:
    return load_config(SMOKE, {"retrieval.index_dir": str(toy_index), **overrides})


def make_trainer(
    cfg: Config,
    *,
    searcher: trainer_module.Searcher,
    reward_mode: ScoringMode = "normal",
    models: LoraPolicyModels | None = None,
    scaler: GradScaler | None = None,
    format_answers: bool = True,
) -> tuple[PPOTrainer, LoraPolicyModels]:
    assert cfg.lora is not None
    models = models or LoraPolicyModels(build_tiny_qwen2(seed=0), cfg.lora, gradient_checkpointing=False)
    engine: RolloutEngine = HFGenerateEngine(models, build_tiny_tokenizer(), cfg.rollout, CPU)
    if format_answers:
        engine = AnswerFormatEngine(engine)
    trainer = PPOTrainer(
        cfg, models=models, engine=engine, searcher=searcher, reward_mode=reward_mode, dist=DIST, scaler=scaler
    )
    return trainer, models


def examples(n: int, relevant: str = "docB") -> list[Example]:
    return [Example(qid=f"q{i}", query="solar efficiency", relevant_ids=frozenset({relevant})) for i in range(n)]


class NanOldLogprobs(LoraPolicyModels):
    """Actor logits are NaN in no-grad forwards (the rollout's old log-probs)."""

    def actor_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        logits = super().actor_logits(input_ids, attention_mask)
        return logits if torch.is_grad_enabled() else torch.full_like(logits, float("nan"))


class NanNewLogprobs(LoraPolicyModels):
    """Actor logits are NaN in grad-enabled forwards (the update's new log-probs)."""

    def actor_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        logits = super().actor_logits(input_ids, attention_mask)
        return logits * float("nan") if torch.is_grad_enabled() else logits


class NanValues(LoraPolicyModels):
    def critic_values(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        values = super().critic_values(input_ids, attention_mask)
        return values if torch.is_grad_enabled() else torch.full_like(values, float("nan"))


def nan_models(cls: type[LoraPolicyModels], cfg: Config) -> LoraPolicyModels:
    assert cfg.lora is not None
    return cls(build_tiny_qwen2(seed=0), cfg.lora, gradient_checkpointing=False)


# --------------------------------------------------------------------------- alignment
def test_response_logprobs_take_the_positions_that_predict_response_tokens() -> None:
    # Prompt of P=3 tokens, response of R=2. Logits at position t predict token t+1, so response
    # token j (absolute position P+j) is scored by the logits at position P-1+j.
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 5, 7, generator=gen)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    expected = torch.stack(
        [torch.log_softmax(logits[0, 2], -1)[4], torch.log_softmax(logits[0, 3], -1)[5]]
    ).unsqueeze(0)
    torch.testing.assert_close(response_logprobs(logits, ids, prompt_len=3), expected)


def test_response_values_are_the_states_before_each_response_token() -> None:
    # values[t] is the value of the state after token t; response token j is chosen in the state
    # after token P-1+j, so the response values are values[P-1 : L-1].
    values = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]])
    torch.testing.assert_close(response_values(values, prompt_len=3), torch.tensor([[12.0, 13.0]]))


def test_response_slices_reject_a_missing_response() -> None:
    with pytest.raises(ValueError, match="no response tokens"):
        response_values(torch.zeros(1, 3), prompt_len=3)


# --------------------------------------------------------------------------- step behaviour
@pytest.mark.java
@pytest.mark.slow
def test_step_returns_finite_metrics_with_expected_keys(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    searcher = Bm25Searcher(toy_index, k1=0.9, b=0.4, threads=2, timeout_s=30.0)
    trainer, _ = make_trainer(cfg, searcher=searcher)
    metrics = trainer.step(examples(cfg.ppo.batch_size), step=0)
    missing = [k for k in [*EVERY_STEP_KEYS, *FORMAT_ERROR_KEYS, *DEEP_KEYS_PRESENT] if k not in metrics]
    assert missing == []
    nonfinite = {k: v for k, v in metrics.items() if not math.isfinite(v)}
    assert nonfinite == {}
    assert -2.0 <= metrics["reward/min"] <= metrics["reward/mean"] <= metrics["reward/max"] <= 1.1
    assert metrics["reward/format_error_rate"] == pytest.approx(0.0)  # every answer is well-formed
    assert metrics["optim/scaler_scale"] == pytest.approx(1.0)  # no GradScaler on fp32


def test_deep_log_keys_only_on_deep_log_steps(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    metrics = trainer.step(examples(cfg.ppo.batch_size), step=1)  # deep_log_every = 10
    assert [k for k in metrics if k.startswith(("stab/", "layers/", "act/"))] == []


def _split_params(models: LoraPolicyModels, base: torch.nn.Module) -> dict[str, dict[str, Tensor]]:
    actor = {id(p) for p in models.actor_parameters()}
    critic = {id(p) for p in models.critic_parameters()}
    groups: dict[str, dict[str, Tensor]] = {"actor": {}, "critic": {}, "base": {}}
    for name, p in base.named_parameters():
        group = "actor" if id(p) in actor else "critic" if id(p) in critic else "base"
        groups[group][name] = p.detach().clone()
    return groups


def test_step_changes_actor_and_critic_not_base(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    assert cfg.lora is not None
    base = build_tiny_qwen2(seed=0)
    models = LoraPolicyModels(base, cfg.lora, gradient_checkpointing=False)
    # Reward 1.1 for every row: returns differ from the zero-initialised values, so the head learns.
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), models=models)
    head_before = [p.detach().clone() for p in models.critic_parameters()[-2:]]
    before = _split_params(models, base)
    trainer.step(examples(cfg.ppo.batch_size, relevant="docN"), step=0)
    after = _split_params(models, base)
    changed = {g: [k for k in before[g] if not torch.equal(before[g][k], after[g][k])] for g in before}
    assert changed["base"] == []
    # LoRA B starts at zero, so A gets no gradient on the first mini-batch and only a ~lr-sized
    # one later (below fp32 resolution for some q/k A matrices); every actor B must move.
    actor_b = sorted(k for k in before["actor"] if ".lora_B." in k)
    assert len(actor_b) == 14
    assert sorted(k for k in changed["actor"] if ".lora_B." in k) == actor_b
    # The value head starts at zero, so the critic LoRA sees ~no gradient in step 1; the head must move.
    head_after = models.critic_parameters()[-2:]
    assert all(not torch.equal(b, a) for b, a in zip(head_before, head_after, strict=True))


@pytest.mark.parametrize(("reward_mode", "reward"), [("normal", 1.1), ("count_once", 0.0)])
def test_reward_uses_requested_mode_and_logs_other(toy_index: Path, reward_mode: ScoringMode, reward: float) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), reward_mode=reward_mode)
    metrics = trainer.step(examples(cfg.ppo.batch_size, relevant="docN"), step=1)
    assert metrics["reward/mean"] == pytest.approx(reward)
    assert metrics["reward/min"] == pytest.approx(reward)
    assert metrics["ndcg/train_normal"] == pytest.approx(1.0)
    assert metrics["ndcg/train_count_once"] == pytest.approx(0.0)


def test_format_errors_get_minus_two_and_are_counted_by_reason(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    # Raw tiny-model text has no tags at all.
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), format_answers=False)
    metrics = trainer.step(examples(cfg.ppo.batch_size), step=1)
    assert metrics["reward/mean"] == pytest.approx(-2.0)
    assert metrics["reward/format_error_rate"] == pytest.approx(1.0)
    assert metrics["reward/format_error/tag_count"] == pytest.approx(1.0)
    assert metrics["reward/format_error/bad_json"] == pytest.approx(0.0)


def test_too_many_clauses_scores_zero_and_is_counted(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=OverflowSearcher())
    metrics = trainer.step(examples(cfg.ppo.batch_size, relevant="docN"), step=1)
    assert metrics["search/too_many_clauses_rate"] == pytest.approx(0.5)
    assert metrics["reward/min"] == pytest.approx(0.0)
    assert metrics["reward/mean"] == pytest.approx(0.55)  # half score 0, half 1.1


def test_step_rejects_wrong_local_batch_size(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    with pytest.raises(ValueError, match="expected 4 examples"):
        trainer.step(examples(3), step=0)


# --------------------------------------------------------------------------- guards
@pytest.mark.parametrize(
    ("cls", "name"),
    [(NanOldLogprobs, "logprobs_old"), (NanNewLogprobs, "logprobs_new"), (NanValues, "values_old")],
)
def test_nonfinite_tensor_raises_naming_tensor_and_step(
    toy_index: Path, cls: type[LoraPolicyModels], name: str
) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), models=nan_models(cls, cfg))
    with pytest.raises(NonFiniteError, match=f"{name} at step 0"):
        trainer.step(examples(cfg.ppo.batch_size), step=0)


def test_nan_logprob_raises_nonfinite_error(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), models=nan_models(NanOldLogprobs, cfg))
    with pytest.raises(NonFiniteError, match="logprobs_old at step 0"):
        trainer.step(examples(cfg.ppo.batch_size), step=0)


def test_nonfinite_advantages_raise(toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    real_gae = trainer_module.gae

    def nan_gae(*args: Tensor, **kwargs: float) -> tuple[Tensor, Tensor]:
        adv, ret = real_gae(*args, **kwargs)
        return adv * float("nan"), ret

    monkeypatch.setattr(trainer_module, "gae", nan_gae)
    with pytest.raises(NonFiniteError, match="advantages at step 3"):
        trainer.step(examples(cfg.ppo.batch_size), step=3)


def test_nonfinite_policy_loss_raises(toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    real = trainer_module.policy_loss

    def nan_loss(*args: Tensor, **kwargs: float) -> tuple[Tensor, dict[str, Tensor]]:
        loss, stats = real(*args, **kwargs)
        return loss * float("nan"), stats

    monkeypatch.setattr(trainer_module, "policy_loss", nan_loss)
    with pytest.raises(NonFiniteError, match="policy_loss at step 2"):
        trainer.step(examples(cfg.ppo.batch_size), step=2)


def test_nonfinite_value_loss_raises(toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    real = trainer_module.value_loss

    def nan_loss(*args: Tensor, **kwargs: float) -> tuple[Tensor, dict[str, Tensor]]:
        loss, stats = real(*args, **kwargs)
        return loss * float("nan"), stats

    monkeypatch.setattr(trainer_module, "value_loss", nan_loss)
    with pytest.raises(NonFiniteError, match="value_loss at step 2"):
        trainer.step(examples(cfg.ppo.batch_size), step=2)


@pytest.mark.parametrize("bad_reward", [1.2, -2.5, float("nan")])
def test_reward_outside_range_raises_assertion(
    toy_index: Path, monkeypatch: pytest.MonkeyPatch, bad_reward: float
) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    monkeypatch.setattr(trainer_module, "compute_reward", lambda *_: bad_reward)
    with pytest.raises(AssertionError, match=r"outside \[-2.0, 1.1\] at step 1"):
        trainer.step(examples(cfg.ppo.batch_size), step=1)


def test_single_bad_reward_in_batch_raises_with_its_value(toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    calls = iter(range(100))
    monkeypatch.setattr(trainer_module, "compute_reward", lambda *_: 7.0 if next(calls) == 0 else 0.0)
    with pytest.raises(AssertionError, match=r"reward 7.0 outside"):
        trainer.step(examples(cfg.ppo.batch_size), step=1)


def test_low_grad_scaler_scale_for_five_steps_raises(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    # Scale 0.5 never grows (growth_interval is huge), so every step counts towards the streak.
    scaler = GradScaler("cpu", init_scale=0.5, growth_interval=10_000)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), scaler=scaler)
    batch = examples(cfg.ppo.batch_size)
    for step in range(1, 5):
        assert trainer.step(batch, step=step)["optim/scaler_scale"] == pytest.approx(0.5)
    with pytest.raises(TrainingHealthError, match=re.escape("scale 0.5 < 1.0 for 5 consecutive steps")):
        trainer.step(batch, step=5)


def test_scale_at_the_floor_is_healthy(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    scaler = GradScaler("cpu", init_scale=1.0, growth_interval=10_000)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), scaler=scaler)
    batch = examples(cfg.ppo.batch_size)
    scales = [trainer.step(batch, step=step)["optim/scaler_scale"] for step in range(1, 7)]
    assert scales == pytest.approx([1.0] * 6)


def test_healthy_scale_resets_the_low_scale_streak(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    scaler = GradScaler("cpu", init_scale=0.5, growth_interval=10_000)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), scaler=scaler)
    batch = examples(cfg.ppo.batch_size)
    for step in range(1, 5):
        trainer.step(batch, step=step)
    scaler.update(new_scale=2.0)
    assert trainer.step(batch, step=5)["optim/scaler_scale"] == pytest.approx(2.0)
    scaler.update(new_scale=0.5)
    for step in range(6, 10):  # four low steps after the reset: still healthy
        trainer.step(batch, step=step)
    with pytest.raises(TrainingHealthError, match="at step 10"):
        trainer.step(batch, step=10)


def test_nonfinite_grad_norm_without_scaler_raises(toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *_: torch.tensor(float("inf")))
    with pytest.raises(NonFiniteError, match="actor_grad_norm at step 1"):
        trainer.step(examples(cfg.ppo.batch_size), step=1)


def test_approx_kl_at_the_bound_is_healthy(toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    real = trainer_module.policy_loss

    def kl_one(*args: Tensor, **kwargs: float) -> tuple[Tensor, dict[str, Tensor]]:
        loss, stats = real(*args, **kwargs)
        return loss, {**stats, "approx_kl": torch.tensor(1.0)}

    monkeypatch.setattr(trainer_module, "policy_loss", kl_one)
    batch = examples(cfg.ppo.batch_size)
    kls = [trainer.step(batch, step=step)["ppo/approx_kl"] for step in range(1, 7)]
    assert kls == pytest.approx([1.0] * 6)


def test_high_approx_kl_for_five_steps_raises(toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    real = trainer_module.policy_loss
    kl = {"value": 2.0}

    def high_kl(*args: Tensor, **kwargs: float) -> tuple[Tensor, dict[str, Tensor]]:
        loss, stats = real(*args, **kwargs)
        return loss, {**stats, "approx_kl": torch.tensor(kl["value"])}

    monkeypatch.setattr(trainer_module, "policy_loss", high_kl)
    batch = examples(cfg.ppo.batch_size)
    for step in range(1, 5):
        trainer.step(batch, step=step)
    kl["value"] = 0.0  # one healthy step resets the streak
    trainer.step(batch, step=5)
    kl["value"] = 2.0
    for step in range(6, 10):
        trainer.step(batch, step=step)
    with pytest.raises(TrainingHealthError, match=re.escape("approx_kl 2.0 > 1.0 for 5 consecutive steps")):
        trainer.step(batch, step=10)


# --------------------------------------------------------------------------- validation and state
def test_validate_scores_both_modes_greedily(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher(), reward_mode="count_once")
    metrics, samples = trainer.validate_with_samples(examples(10, relevant="docN"))
    assert metrics == {"val/ndcg_normal": pytest.approx(1.0), "val/ndcg_count_once": pytest.approx(0.0)}
    assert len(samples) == 8
    assert samples[0].reward_normal == pytest.approx(1.1)
    assert samples[0].reward_count_once == pytest.approx(0.0)
    assert trainer.validate(examples(3, relevant="docC")) == {
        "val/ndcg_normal": pytest.approx(0.0),
        "val/ndcg_count_once": pytest.approx(1.0),
    }


def _exp_avgs(state: dict[str, Any]) -> list[Tensor]:
    return [state["state"][key]["exp_avg"] for key in sorted(state["state"])]


@pytest.mark.parametrize("group", ["actor", "critic"])
def test_optimizer_states_round_trip(toy_index: Path, group: str) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    trainer.step(examples(cfg.ppo.batch_size), step=1)
    states = trainer.optimizer_states()
    assert sorted(states) == ["actor", "critic"]
    fresh, _ = make_trainer(cfg, searcher=ModeSearcher())
    fresh.load_optimizer_states(states)
    loaded = fresh.optimizer_states()[group]
    assert loaded["state"].keys() == states[group]["state"].keys()
    torch.testing.assert_close(_exp_avgs(loaded), _exp_avgs(states[group]))


def test_load_optimizer_states_rejects_missing_group(toy_index: Path) -> None:
    cfg = smoke_cfg(toy_index)
    trainer, _ = make_trainer(cfg, searcher=ModeSearcher())
    with pytest.raises(KeyError, match="critic"):
        trainer.load_optimizer_states({"actor": trainer.optimizer_states()["actor"]})


# --------------------------------------------------------------------------- learning
@pytest.mark.slow
def test_ppo_learns_toy_reward(toy_index: Path) -> None:
    """Reward 1.1 iff the response contains token id 7 (``w4``): PPO must find and keep it."""
    cfg = smoke_cfg(
        toy_index,
        **{
            "ppo.actor_lr": "1e-2",
            "ppo.critic_lr": "1e-2",
            "ppo.batch_size": "16",
            "ppo.mini_batch_size": "8",
            "ppo.micro_batch_size": "4",
            "rollout.max_new_tokens": "8",
            "rollout.generation_batch_size": "16",
            "run.deep_log_every": "1000",
        },
    )
    assert cfg.lora is not None
    base = build_tiny_qwen2(seed=0)
    # The tied embedding is frozen under LoRA and initialised with std 0.02, which caps every
    # logit near |0.84| (Task 9): token 7 can never become likely. x5 lifts that cap; measured at
    # scale 1 the reward stays near 0.2 after 30 steps, at scale 5 it reaches 1.1 by step ~10.
    embedding = base.get_input_embeddings()
    assert isinstance(embedding, torch.nn.Embedding)
    with torch.no_grad():
        embedding.weight.mul_(5.0)
    models = LoraPolicyModels(base, cfg.lora, gradient_checkpointing=False)
    trainer, _ = make_trainer(cfg, searcher=TokenSevenSearcher(), models=models)
    batch = examples(cfg.ppo.batch_size, relevant="hit")
    rewards = [trainer.step(batch, step=s)["reward/mean"] for s in range(1, 31)]
    # Step 1's rollout comes from the untrained policy, so its reward is the baseline.
    assert rewards[0] < 0.2, rewards
    assert sum(rewards[-5:]) / 5 > 0.6, rewards
