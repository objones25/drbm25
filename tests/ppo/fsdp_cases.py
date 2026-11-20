"""Two-rank FSDP2 cases for ``test_models_full.py``, run as a script (not collected by pytest).

    python fsdp_cases.py CASE OUT_DIR

``torch.multiprocessing.spawn`` starts two gloo ranks on CPU with the tiny fp32 Qwen2; rank 0
writes the case's tensors to ``OUT_DIR/result.pt``. The test process compares them with a
single-process reference built from the same helpers (imported from this file by path).
"""

from __future__ import annotations

import dataclasses
import os
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch
import torch.distributed as tdist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.multiprocessing.spawn import spawn

from drbm25.bm25.searcher import SearchResult
from drbm25.checkpoint import TrainState, latest_checkpoint, load_sharded_checkpoint, save_sharded_checkpoint
from drbm25.config import Config, ScoringMode, load_config
from drbm25.data import Example
from drbm25.distributed import DistInfo
from drbm25.ppo.models import PolicyModels, token_logprobs
from drbm25.ppo.models_full import FullPolicyModels
from drbm25.ppo.trainer import PPOTrainer
from drbm25.rollout.base import RolloutBatch, encode_prompts, finalize_rollout, special_ids
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer

WORLD: Final = 2
CPU: Final = torch.device("cpu")
LR: Final = 1e-3
SMOKE: Final = Path(__file__).resolve().parents[2] / "configs" / "smoke_cpu.yaml"


# ----------------------------------------------------------------------------- shared helpers
def build_models(mesh: DeviceMesh, *, seed: int = 0, gradient_checkpointing: bool = False) -> FullPolicyModels:
    return FullPolicyModels(
        build_tiny_qwen2(seed),
        build_tiny_qwen2(seed).model,
        build_tiny_qwen2(seed),
        mesh=mesh,
        compute_dtype=torch.float32,
        gradient_checkpointing=gradient_checkpointing,
    )


def lm_batch() -> tuple[Tensor, Tensor]:
    """4 rows x 8 tokens, the same on every rank (fixed generator)."""
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(3, 128, (4, 8), generator=gen)
    return ids, torch.ones_like(ids)


def lm_loss(logits: Tensor, ids: Tensor) -> Tensor:
    return -token_logprobs(logits, ids).mean()


def full_cfg() -> Config:
    """Smoke config switched to the full layout: local batch 2 per rank on 2 ranks."""
    cfg = load_config(SMOKE, {"model.train_mode": "full", "run.deep_log_every": "1"})
    return dataclasses.replace(cfg, lora=None)


def same_examples(n: int) -> list[Example]:
    return [Example(qid=f"q{i}", query="solar efficiency", relevant_ids=frozenset({"hit"})) for i in range(n)]


class FixedEngine:
    """Every row answers ``w4 w5`` then EOS (no sampling), in the answer format the reward parses."""

    def __init__(self) -> None:
        self._tokenizer = build_tiny_tokenizer()
        self._ids = special_ids(self._tokenizer)
        self.synced = 0

    def generate(self, prompts: Sequence[str], *, greedy: bool) -> RolloutBatch:
        del greedy
        prompt_ids, prompt_mask = encode_prompts(self._tokenizer, prompts, CPU, max_prompt_tokens=256)
        row = self._tokenizer("w4 w5").data["input_ids"] + [self._ids.eos]
        response = torch.tensor([row] * len(prompts))
        batch = finalize_rollout(prompt_ids, prompt_mask, response, tokenizer=self._tokenizer, ids=self._ids)
        texts = tuple(f'</think>\n<answer>{{"query": "{t}"}}</answer>' for t in batch.response_text)
        return dataclasses.replace(batch, response_text=texts)

    def sync_weights(self, models: PolicyModels) -> None:
        models.merged_actor_state_dict()  # collective on FSDP: every rank must take part
        self.synced += 1


class HitSearcher:
    """``hit`` (relevant) ranks first iff the query contains ``w4``."""

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult:
        del mode, k
        words = query.split()
        return SearchResult(("hit" if "w4" in words else "miss",), len(words), len(set(words)), (1.0,))

    def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]:
        return [self.search(q, mode=mode, k=k) for q in queries]

    def analyze(self, text: str) -> list[str]:
        return text.split()


def ppo_step(mesh: DeviceMesh, dist: DistInfo, cfg: Config) -> tuple[FullPolicyModels, dict[str, float]]:
    """One PPOTrainer step on the full layout; every rank gets the same 2 examples."""
    torch.manual_seed(0)
    models = build_models(mesh)
    trainer = PPOTrainer(
        cfg, models=models, engine=FixedEngine(), searcher=HitSearcher(), reward_mode="normal", dist=dist, scaler=None
    )
    metrics = trainer.step(same_examples(cfg.ppo.batch_size // dist.world_size), step=0)
    return models, metrics


def gathered_critic(models: FullPolicyModels) -> dict[str, Tensor]:
    return {k[len("critic.") :]: v for k, v in models.trainable_state_dict().items() if k.startswith("critic.")}


# ----------------------------------------------------------------------------- cases
def case_actor_step(rank: int, mesh: DeviceMesh, out: Path) -> None:
    """Each rank trains on its half of the batch; FSDP must average to the full-batch step."""
    models = build_models(mesh, gradient_checkpointing=True)
    params = models.actor_parameters()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    ids, attn = lm_batch()
    rows = slice(2 * rank, 2 * rank + 2)
    for _ in range(2):
        lm_loss(models.actor_logits(ids[rows], attn[rows]), ids[rows]).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    merged = models.merged_actor_state_dict()
    if rank == 0:
        torch.save({k: v.clone() for k, v in merged.items()}, out / "result.pt")


def case_checkpoint(rank: int, mesh: DeviceMesh, out: Path) -> None:
    """Train one step, save sharded; a fresh model + optimizer loads it back exactly."""
    dist = DistInfo(rank, WORLD, rank, CPU)
    run_dir = out / "run"
    trained = build_models(mesh, seed=0)
    opts = {
        "actor": torch.optim.AdamW(trained.actor_parameters(), lr=LR, weight_decay=0.0),
        "critic": torch.optim.AdamW(trained.critic_parameters(), lr=LR, weight_decay=0.0),
    }
    ids, attn = lm_batch()
    lm_loss(trained.actor_logits(ids, attn), ids).backward()
    (trained.critic_values(ids, attn) - 1.0).pow(2).mean().backward()  # nonzero grad at a zero head
    for opt in opts.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    state = TrainState(step=3, data_cursor=12, wandb_run_id="abc", reward_mode="count_once")
    save_sharded_checkpoint(run_dir, train_state=state, state=trained.sharded_state(opts), keep=2, dist=dist)

    fresh = build_models(mesh, seed=1)
    fresh_opts = {
        "actor": torch.optim.AdamW(fresh.actor_parameters(), lr=LR, weight_decay=0.0),
        "critic": torch.optim.AdamW(fresh.critic_parameters(), lr=LR, weight_decay=0.0),
    }
    path = latest_checkpoint(run_dir)
    if path is None:
        raise RuntimeError("no complete checkpoint after save_sharded_checkpoint")
    template = fresh.sharded_state(fresh_opts)
    loaded = load_sharded_checkpoint(path, template)
    fresh.load_sharded_state(template, fresh_opts)

    want, got = trained.trainable_state_dict(), fresh.trainable_state_dict()
    exp_avg = {
        name: [
            s["exp_avg"].full_tensor()
            for s in (opts[name].state[p] for p in group)  # pyright: ignore[reportAttributeAccessIssue]
        ]
        for name, group in (("actor", trained.actor_parameters()), ("critic", trained.critic_parameters()))
    }
    fresh_exp_avg = {
        name: [s["exp_avg"].full_tensor() for s in (fresh_opts[name].state[p] for p in group)]  # pyright: ignore[reportAttributeAccessIssue]
        for name, group in (("actor", fresh.actor_parameters()), ("critic", fresh.critic_parameters()))
    }
    if rank == 0:
        torch.save(
            {
                "want": want,
                "got": got,
                "exp_avg": exp_avg,
                "fresh_exp_avg": fresh_exp_avg,
                "train_state": dataclasses.asdict(loaded.train_state),
                "files": sorted(p.name for p in path.iterdir()),
                "path": str(path),
            },
            out / "result.pt",
        )


def case_ppo_step(rank: int, mesh: DeviceMesh, out: Path) -> None:
    models, metrics = ppo_step(mesh, DistInfo(rank, WORLD, rank, CPU), full_cfg())
    actor = {k: v.clone() for k, v in models.merged_actor_state_dict().items()}
    critic = gathered_critic(models)
    if rank == 0:
        torch.save({"actor": actor, "critic": critic, "metrics": metrics}, out / "result.pt")


CASES: Final = {"actor_step": case_actor_step, "checkpoint": case_checkpoint, "ppo_step": case_ppo_step}


def _worker(rank: int, case: str, port: int, out: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    tdist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        CASES[case](rank, init_device_mesh("cpu", (WORLD,)), Path(out))
    finally:
        tdist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


if __name__ == "__main__":
    case_name, out_dir = sys.argv[1], sys.argv[2]
    if case_name not in CASES:
        raise SystemExit(f"unknown case {case_name!r}; expected one of {sorted(CASES)}")
    spawn(_worker, args=(case_name, _free_port(), out_dir), nprocs=WORLD, join=True)
