"""FullPolicyModels: the A100 layout (FSDP2 full fine-tune) on CPU/gloo with the tiny fp32 Qwen2.

In-process tests use a one-rank gloo group (FSDP2 needs a process group even for one rank).
Two-rank tests run ``tests/ppo/fsdp_cases.py`` in a subprocess, which starts two gloo ranks with
``torch.multiprocessing.spawn``, and compare its tensors with a single-process reference.
"""

import importlib.util
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import torch.distributed as tdist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from torch.testing import assert_close
from transformers import Qwen2Config, Qwen2ForCausalLM

from drbm25.checkpoint import (
    TrainState,
    latest_checkpoint,
    load_checkpoint,
    load_sharded_actor,
    load_sharded_checkpoint,
    save_sharded_checkpoint,
)
from drbm25.distributed import DistInfo
from drbm25.ppo.models import PolicyModels
from drbm25.ppo.models_full import FullPolicyModels
from drbm25.tiny import build_tiny_qwen2

CASES_FILE = Path(__file__).with_name("fsdp_cases.py")
SRC = Path(__file__).resolve().parents[2] / "src"
CPU = torch.device("cpu")
ONE_RANK = DistInfo(rank=0, world_size=1, local_rank=0, device=CPU)
# DCP's filesystem writer calls the deprecated tensor.storage() (torch 2.10); not our code.
DCP_WARNING = "ignore:TypedStorage is deprecated:UserWarning"


def _load_cases() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fsdp_cases", CASES_FILE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()


@pytest.fixture
def mesh(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeviceMesh]:
    store = tmp_path_factory.mktemp("pg") / "store"
    tdist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        tdist.destroy_process_group()


def _run_two_ranks_per_rank(case: str, out: Path) -> list[dict[str, Any]]:
    _run_case(case, out)
    return [torch.load(out / f"result_rank{r}.pt", weights_only=True) for r in range(2)]


def _run_case(case: str, out: Path) -> None:
    proc = subprocess.run(  # noqa: S603 -- this interpreter and a script in this repo
        [sys.executable, str(CASES_FILE), case, str(out)],
        env={**os.environ, "PYTHONPATH": str(SRC), "HF_HUB_OFFLINE": "1"},  # JAVA_HOME: the trainer imports pyserini
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]


def _run_two_ranks(case: str, out: Path) -> dict[str, Any]:
    proc = subprocess.run(  # noqa: S603 -- this interpreter and a script in this repo
        [sys.executable, str(CASES_FILE), case, str(out)],
        env={**os.environ, "PYTHONPATH": str(SRC), "HF_HUB_OFFLINE": "1"},  # JAVA_HOME: the trainer imports pyserini
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return torch.load(out / "result.pt", weights_only=True)


# --------------------------------------------------------------------------- protocol and forwards
def test_full_models_match_lora_protocol(mesh: DeviceMesh) -> None:
    models = cases.build_models(mesh)
    assert isinstance(models, PolicyModels)
    actor = {id(p) for p in models.actor_parameters()}
    critic = {id(p) for p in models.critic_parameters()}
    assert actor
    assert critic
    assert not actor & critic
    assert all(isinstance(p, DTensor) for p in models.actor_parameters() + models.critic_parameters())
    assert len(models.decoder_layers()) == 2


def test_forwards_shapes_and_initial_values(mesh: DeviceMesh) -> None:
    models = cases.build_models(mesh)
    ids, attn = cases.lm_batch()
    actor = models.actor_logits(ids, attn)
    ref = models.ref_logits(ids, attn)
    values = models.critic_values(ids, attn)
    assert actor.shape == (4, 8, 128)
    assert not ref.requires_grad
    assert_close(ref, actor.detach())  # same seed: the frozen reference starts equal to the actor
    assert values.shape == (4, 8)
    assert values.dtype == torch.float32
    assert torch.count_nonzero(values) == 0  # zero-initialised value head


def test_actor_logits_match_unsharded_model(mesh: DeviceMesh) -> None:
    models = cases.build_models(mesh)
    ids, attn = cases.lm_batch()
    plain = build_tiny_qwen2(0)
    with torch.no_grad():
        want = plain(input_ids=ids, attention_mask=attn, use_cache=False).logits
        got = models.actor_logits(ids, attn)
    assert_close(got, want)


def test_ref_model_has_no_trainable_parameters(mesh: DeviceMesh) -> None:
    models = cases.build_models(mesh)
    ids, attn = cases.lm_batch()
    before = models.ref_logits(ids, attn)
    loss = cases.lm_loss(models.actor_logits(ids, attn), ids)
    loss.backward()
    opt = torch.optim.AdamW(models.actor_parameters(), lr=1e-2)
    opt.step()
    assert_close(models.ref_logits(ids, attn), before)
    assert not torch.equal(models.actor_logits(ids, attn).detach(), before)


def test_merged_actor_state_dict_names_and_values_match_plain_model(mesh: DeviceMesh) -> None:
    models = cases.build_models(mesh)
    merged = models.merged_actor_state_dict()
    plain = build_tiny_qwen2(0).state_dict()
    assert sorted(merged) == sorted(plain)
    for name, tensor in plain.items():
        assert not isinstance(merged[name], DTensor), name
        assert merged[name].dtype == torch.float32
        assert_close(merged[name], tensor, msg=name)


def test_merged_actor_state_dict_casts_to_compute_dtype(mesh: DeviceMesh) -> None:
    models = FullPolicyModels(
        build_tiny_qwen2(0),
        build_tiny_qwen2(0).model,
        build_tiny_qwen2(0),
        mesh=mesh,
        compute_dtype=torch.bfloat16,
        gradient_checkpointing=False,
    )
    merged = models.merged_actor_state_dict()
    assert {t.dtype for t in merged.values()} == {torch.bfloat16}
    # Master weights stay fp32: only the gathered copy is cast.
    assert {p.dtype for p in models.actor_parameters()} == {torch.float32}


def test_merged_actor_state_dict_can_gather_onto_another_device(mesh: DeviceMesh) -> None:
    # train.py gathers the best policy straight to host memory, one tensor at a time.
    merged = cases.build_models(mesh).merged_actor_state_dict(device=torch.device("meta"))
    assert {t.device.type for t in merged.values()} == {"meta"}


def test_trainable_state_dict_roundtrip(mesh: DeviceMesh) -> None:
    trained = cases.build_models(mesh, seed=0)
    fresh = cases.build_models(mesh, seed=1)
    state = trained.trainable_state_dict()
    assert {k.split(".", 1)[0] for k in state} == {"actor", "critic"}
    assert "critic.value_head.weight" in state
    fresh.load_trainable_state_dict(state)
    for name, tensor in fresh.trainable_state_dict().items():
        assert_close(tensor, state[name], msg=name)


def test_load_trainable_state_dict_rejects_missing_keys(mesh: DeviceMesh) -> None:
    models = cases.build_models(mesh)
    state = models.trainable_state_dict()
    del state["critic.value_head.weight"]
    with pytest.raises(KeyError, match=r"critic\.value_head\.weight"):
        models.load_trainable_state_dict(state)


def test_rejects_critic_with_lm_head(mesh: DeviceMesh) -> None:
    with pytest.raises(TypeError, match="backbone"):
        FullPolicyModels(
            build_tiny_qwen2(0),
            build_tiny_qwen2(0),
            build_tiny_qwen2(0),
            mesh=mesh,
            compute_dtype=torch.float32,
            gradient_checkpointing=False,
        )


def test_discard_pending_is_a_noop(mesh: DeviceMesh) -> None:
    models = cases.build_models(mesh)
    models.discard_pending()
    assert models.generation_module() is models.generation_module()


# --------------------------------------------------------------------------- two ranks
@pytest.mark.slow
def test_fsdp_actor_step_matches_single_process_step(tmp_path: Path) -> None:
    got = _run_two_ranks("actor_step", tmp_path)
    model = build_tiny_qwen2(0)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cases.LR, weight_decay=0.0)
    ids, attn = cases.lm_batch()
    for _ in range(2):
        cases.lm_loss(model(input_ids=ids, attention_mask=attn, use_cache=False).logits, ids).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    want = model.state_dict()
    assert sorted(got) == sorted(want)
    moved = 0
    for name, tensor in want.items():
        assert_close(got[name], tensor, rtol=1e-5, atol=1e-6, msg=name)
        moved += int(not torch.equal(tensor, build_tiny_qwen2(0).state_dict()[name]))
    assert moved == len(want)  # every tensor took a step, so the comparison is not vacuous


@pytest.mark.slow
def test_sharded_checkpoint_roundtrip(tmp_path: Path) -> None:
    got = _run_two_ranks("checkpoint", tmp_path)
    for name, tensor in got["want"].items():
        assert_close(got["got"][name], tensor, msg=name)
    # Adam first moments of both optimizers came back, and they are not trivially zero.
    assert {k: len(v) for k, v in got["fresh_exp_avg"].items()} == {k: len(v) for k, v in got["exp_avg"].items()}
    assert_close(got["fresh_exp_avg"], got["exp_avg"])
    assert {k: any(bool(m.abs().sum() > 0) for m in v) for k, v in got["exp_avg"].items()} == {
        "actor": True,
        "critic": True,
    }
    assert got["train_state"] == {"step": 3, "data_cursor": 12, "wandb_run_id": "abc", "reward_mode": "count_once"}
    assert got["files"] == ["dcp", "meta.json", "rng.pt"]
    run_dir = Path(got["path"]).parent
    assert (run_dir / "latest").read_text().strip() == "step_000003"
    assert sorted(p.name for p in run_dir.iterdir()) == ["latest", "step_000003"]  # no temp dirs left


@pytest.mark.slow
@pytest.mark.filterwarnings(DCP_WARNING)
def test_ppo_step_on_two_ranks_matches_one_rank(tmp_path: Path, mesh: DeviceMesh) -> None:
    """Both ranks see the same examples, so the FSDP-averaged step equals a one-rank step.

    A second, manual all-reduce of FSDP's sharded grads would sum different parameter shards
    and break this equality.
    """
    got = _run_two_ranks("ppo_step", tmp_path)
    cfg = cases.full_cfg()
    one_rank_cfg = cases.dataclasses.replace(
        cfg,
        ppo=cases.dataclasses.replace(
            cfg.ppo, batch_size=cfg.ppo.batch_size // 2, mini_batch_size=cfg.ppo.mini_batch_size // 2
        ),
    )
    models, metrics = cases.ppo_step(mesh, ONE_RANK, one_rank_cfg)
    for name, tensor in models.merged_actor_state_dict().items():
        assert_close(got["actor"][name], tensor, rtol=1e-5, atol=1e-6, msg=name)
    for name, tensor in cases.gathered_critic(models).items():
        assert_close(got["critic"][name], tensor, rtol=1e-5, atol=1e-6, msg=name)
    assert got["metrics"]["optim/actor_update_ratio"] == pytest.approx(metrics["optim/actor_update_ratio"], rel=1e-4)
    assert metrics["optim/actor_update_ratio"] > 0


# --------------------------------------------------------------------------- one rank: trainer and checkpoint
@pytest.mark.filterwarnings(DCP_WARNING)
def test_ppo_step_full_layout_logs_actor_update_ratio_only_on_deep_steps(mesh: DeviceMesh) -> None:
    cfg = cases.full_cfg()
    one_rank_cfg = cases.dataclasses.replace(
        cfg,
        ppo=cases.dataclasses.replace(cfg.ppo, batch_size=2, mini_batch_size=2),
        run=cases.dataclasses.replace(cfg.run, deep_log_every=2),
    )
    torch.manual_seed(0)
    models = cases.build_models(mesh)
    engine = cases.FixedEngine()
    trainer = cases.PPOTrainer(
        one_rank_cfg,
        models=models,
        engine=engine,
        searcher=cases.HitSearcher(),
        reward_mode="normal",
        dist=ONE_RANK,
        scaler=None,
    )
    deep = trainer.step(cases.same_examples(2), step=0)
    shallow = trainer.step(cases.same_examples(2), step=1)
    assert deep["optim/actor_update_ratio"] > 0
    assert math.isfinite(deep["optim/actor_update_ratio"])
    assert "optim/critic_update_ratio" not in deep  # skipped on full FT: see PPOTrainer docstring
    assert "optim/actor_update_ratio" not in shallow
    assert "stab/qk_logit_bound_max" in deep
    assert engine.synced == 2
    assert all(p.grad is None for p in models.actor_parameters() + models.critic_parameters())


@pytest.mark.filterwarnings(DCP_WARNING)
def test_sharded_checkpoint_one_rank_roundtrip_and_prune(mesh: DeviceMesh, tmp_path: Path) -> None:
    models = cases.build_models(mesh)
    opts = {
        "actor": torch.optim.AdamW(models.actor_parameters(), lr=cases.LR),
        "critic": torch.optim.AdamW(models.critic_parameters(), lr=cases.LR),
    }
    for step in range(1, 4):
        state = TrainState(step=step, data_cursor=4 * step, wandb_run_id=None, reward_mode="normal")
        save_sharded_checkpoint(tmp_path, train_state=state, state=models.sharded_state(opts), keep=2, dist=ONE_RANK)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["latest", "step_000002", "step_000003"]
    path = latest_checkpoint(tmp_path)
    assert path is not None
    assert path.name == "step_000003"
    loaded = load_sharded_checkpoint(path, models.sharded_state(opts))
    assert loaded.train_state == TrainState(step=3, data_cursor=12, wandb_run_id=None, reward_mode="normal")
    assert set(loaded.rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}


@pytest.mark.filterwarnings(DCP_WARNING)
def test_load_checkpoint_refuses_sharded_dir(mesh: DeviceMesh, tmp_path: Path) -> None:
    models = cases.build_models(mesh)
    opts = {
        "actor": torch.optim.AdamW(models.actor_parameters()),
        "critic": torch.optim.AdamW(models.critic_parameters()),
    }
    state = TrainState(step=1, data_cursor=4, wandb_run_id=None, reward_mode="normal")
    path = save_sharded_checkpoint(tmp_path, train_state=state, state=models.sharded_state(opts), keep=1, dist=ONE_RANK)
    with pytest.raises(ValueError, match="sharded"):
        load_checkpoint(path)


def test_load_sharded_checkpoint_refuses_single_file_dir(tmp_path: Path) -> None:
    from drbm25.checkpoint import save_checkpoint

    state = TrainState(step=1, data_cursor=4, wandb_run_id=None, reward_mode="normal")
    path = save_checkpoint(tmp_path, train_state=state, model={"w": torch.zeros(2)}, optimizers={}, scaler=None, keep=1)
    with pytest.raises(ValueError, match="not a sharded"):
        load_sharded_checkpoint(path, {})


# --------------------------------------------------------------------------- LoRA layout: same weights on every rank
@pytest.mark.slow
def test_lora_construction_gives_every_rank_the_same_weights(tmp_path: Path) -> None:
    """Fails when the per-rank seed is applied before the adapters are initialised (broadcast disabled)."""
    rank0, rank1 = _run_two_ranks_per_rank("lora_init_construction", tmp_path)
    assert sorted(rank0["init"]) == sorted(rank1["init"])
    assert_close(rank1["init"], rank0["init"], rtol=0, atol=0)


@pytest.mark.slow
def test_lora_trainable_params_bit_identical_across_ranks_after_init_and_step(tmp_path: Path) -> None:
    rank0, rank1 = _run_two_ranks_per_rank("lora_init", tmp_path)
    assert_close(rank1["init"], rank0["init"], rtol=0, atol=0)
    assert_close(rank1["stepped"], rank0["stepped"], rtol=0, atol=0)
    moved = [k for k in rank0["init"] if not torch.equal(rank0["init"][k], rank0["stepped"][k])]
    assert len(moved) > len(rank0["init"]) // 2  # the step really changed the adapters


@pytest.mark.slow
def test_lora_validation_is_sharded_across_ranks_and_matches_one_process(tmp_path: Path) -> None:
    single, seen_single = cases.lora_validate(ONE_RANK)
    assert seen_single == 7
    assert single["val/ndcg_normal"] == pytest.approx(3 / 7)
    rank0, rank1 = _run_two_ranks_per_rank("lora_validate", tmp_path)
    assert (rank0["seen"], rank1["seen"]) == (4, 3)  # each rank generates only its slice (rank::2)
    assert rank0["metrics"] == pytest.approx(single)
    assert rank1["metrics"] == pytest.approx(single)


@pytest.mark.slow
def test_full_vllm_validation_is_sharded_so_per_rank_engines_cannot_disagree(tmp_path: Path) -> None:
    """vLLM engines are independent per rank; validation must shard and all-reduce, not compare."""
    rank0, rank1 = _run_two_ranks_per_rank("full_vllm_validate", tmp_path)
    assert (rank0["seen"], rank1["seen"]) == (4, 3)  # each rank generates only its slice (rank::2)
    # rank 0's slice (0, 2, 4, 6) hits at 0 and 4; rank 1 answers w4 on its 3 rows: 5 of 7.
    for result in (rank0, rank1):
        assert result["message"] == ""
        assert result["metrics"]["val/ndcg_normal"] == pytest.approx(5 / 7)
        assert result["metrics"]["val/ndcg_count_once"] == pytest.approx(5 / 7)


def test_assert_params_equal_raises_on_mismatch_and_broadcast_fixes_it(tmp_path: Path) -> None:
    results = _run_two_ranks_per_rank("assert_params", tmp_path)
    assert [r["message"] for r in results] == ["ranks hold different values for parameters at indices [0] (of 2)"] * 2
    assert [r["after"].tolist() for r in results] == [[0.0, 0.0, 0.0]] * 2  # rank 0's values everywhere


@pytest.mark.filterwarnings(DCP_WARNING)
def test_load_sharded_actor_into_plain_model(mesh: DeviceMesh, tmp_path: Path) -> None:
    models = cases.build_models(mesh, seed=0)
    opts = {
        "actor": torch.optim.AdamW(models.actor_parameters()),
        "critic": torch.optim.AdamW(models.critic_parameters()),
    }
    state = TrainState(step=1, data_cursor=4, wandb_run_id=None, reward_mode="count_once")
    path = save_sharded_checkpoint(tmp_path, train_state=state, state=models.sharded_state(opts), keep=1, dist=ONE_RANK)
    plain = build_tiny_qwen2(seed=99)  # different init: equality below proves the load
    assert load_sharded_actor(path, plain) == state
    assert_close(plain.state_dict(), models.merged_actor_state_dict(), rtol=0, atol=0)
    one_layer = Qwen2ForCausalLM(Qwen2Config(**{**plain.config.to_dict(), "num_hidden_layers": 1, "layer_types": None}))
    with pytest.raises(KeyError, match=r"unexpected=\['model\.layers\.1\."):
        load_sharded_actor(path, one_layer)
