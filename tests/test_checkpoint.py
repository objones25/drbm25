import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from drbm25.checkpoint import (
    LoadedCheckpoint,
    TrainState,
    capture_rng,
    latest_checkpoint,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)


def _model_and_optim() -> tuple[nn.Linear, torch.optim.AdamW]:
    model = nn.Linear(3, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 3)).sum().backward()
    optim.step()
    return model, optim


def _save(run_dir: Path, step: int, *, keep: int = 5) -> Path:
    model, optim = _model_and_optim()
    return save_checkpoint(
        run_dir,
        train_state=TrainState(step=step, data_cursor=step * 2, wandb_run_id="abc", reward_mode="normal"),
        model=model.state_dict(),
        optimizers={"actor": optim.state_dict()},
        scaler=None,
        keep=keep,
    )


def test_roundtrip_restores_everything(tmp_path: Path) -> None:
    model, optim = _model_and_optim()
    scaler_state = {"scale": 1024.0, "growth_factor": 2.0, "_growth_tracker": 3}
    path = save_checkpoint(
        tmp_path,
        train_state=TrainState(step=50, data_cursor=800, wandb_run_id="run-1", reward_mode="normal"),
        model=model.state_dict(),
        optimizers={"actor": optim.state_dict(), "critic": optim.state_dict()},
        scaler=scaler_state,
        keep=3,
    )
    assert path == tmp_path / "step_000050"

    loaded = load_checkpoint(path)

    assert isinstance(loaded, LoadedCheckpoint)
    assert loaded.path == path
    assert loaded.train_state == TrainState(step=50, data_cursor=800, wandb_run_id="run-1", reward_mode="normal")
    assert set(loaded.model) == {"weight", "bias"}
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.model[name], tensor)
    assert set(loaded.optimizers) == {"actor", "critic"}
    saved_state = optim.state_dict()["state"][0]
    got_state = loaded.optimizers["actor"]["state"][0]
    assert torch.equal(got_state["exp_avg"], saved_state["exp_avg"])
    assert torch.equal(got_state["step"], saved_state["step"])
    assert loaded.scaler == scaler_state
    # the loaded state must be accepted by a fresh optimizer
    fresh = nn.Linear(3, 2)
    fresh_optim = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_optim.load_state_dict(loaded.optimizers["actor"])


def test_roundtrip_with_none_scaler_and_none_run_id(tmp_path: Path) -> None:
    model, optim = _model_and_optim()
    path = save_checkpoint(
        tmp_path,
        train_state=TrainState(step=1, data_cursor=0, wandb_run_id=None, reward_mode="normal"),
        model=model.state_dict(),
        optimizers={"actor": optim.state_dict()},
        scaler=None,
        keep=1,
    )
    loaded = load_checkpoint(path)
    assert loaded.scaler is None
    assert loaded.train_state.wandb_run_id is None


def test_saves_shared_storage_tensors(tmp_path: Path) -> None:
    weight = torch.randn(2, 2)
    path = save_checkpoint(
        tmp_path,
        train_state=TrainState(step=1, data_cursor=0, wandb_run_id=None, reward_mode="normal"),
        model={"a": weight, "b": weight, "c": weight.t()},
        optimizers={},
        scaler=None,
        keep=1,
    )
    loaded = load_checkpoint(path)
    assert torch.equal(loaded.model["a"], weight)
    assert torch.equal(loaded.model["b"], weight)
    assert torch.equal(loaded.model["c"], weight.t())


def test_latest_points_to_newest(tmp_path: Path) -> None:
    _save(tmp_path, 50)
    _save(tmp_path, 100)
    assert latest_checkpoint(tmp_path) == tmp_path / "step_000100"
    assert (tmp_path / "latest").read_text().strip() == "step_000100"


def test_latest_none_for_empty_or_missing_dir(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "nope") is None


def test_ignores_incomplete_tmp_dir(tmp_path: Path) -> None:
    _save(tmp_path, 50)
    tmp = tmp_path / ".tmp-step_000100-1"
    tmp.mkdir()
    (tmp / "model.safetensors").write_bytes(b"partial")
    assert latest_checkpoint(tmp_path) == tmp_path / "step_000050"


def test_latest_falls_back_when_pointer_dir_is_incomplete(tmp_path: Path) -> None:
    _save(tmp_path, 50)
    _save(tmp_path, 100)
    (tmp_path / "step_000100" / "meta.json").unlink()
    assert latest_checkpoint(tmp_path) == tmp_path / "step_000050"


def test_latest_falls_back_when_pointer_names_missing_dir(tmp_path: Path) -> None:
    _save(tmp_path, 50)
    (tmp_path / "latest").write_text("step_000999\n")
    assert latest_checkpoint(tmp_path) == tmp_path / "step_000050"


def test_latest_without_pointer_picks_newest_complete(tmp_path: Path) -> None:
    _save(tmp_path, 50)
    _save(tmp_path, 100)
    (tmp_path / "latest").unlink()
    assert latest_checkpoint(tmp_path) == tmp_path / "step_000100"


def test_latest_pointer_wins_over_newer_dir(tmp_path: Path) -> None:
    _save(tmp_path, 50)
    _save(tmp_path, 100)
    (tmp_path / "latest").write_text("step_000050\n")
    assert latest_checkpoint(tmp_path) == tmp_path / "step_000050"


def test_prunes_to_keep(tmp_path: Path) -> None:
    _save(tmp_path, 50, keep=2)
    _save(tmp_path, 100, keep=2)
    _save(tmp_path, 150, keep=2)
    names = sorted(p.name for p in tmp_path.glob("step_*"))
    assert names == ["step_000100", "step_000150"]


def test_prune_never_deletes_latest_target(tmp_path: Path) -> None:
    _save(tmp_path, 100)
    _save(tmp_path, 150)
    # a resumed run saves an older step number; latest then points at it
    _save(tmp_path, 50, keep=2)
    names = sorted(p.name for p in tmp_path.glob("step_*"))
    assert names == ["step_000050", "step_000150"]  # protected dir + (keep - 1) newest others
    assert latest_checkpoint(tmp_path) == tmp_path / "step_000050"


def test_prune_total_never_exceeds_keep(tmp_path: Path) -> None:
    _save(tmp_path, 100)
    _save(tmp_path, 150)
    _save(tmp_path, 50, keep=1)
    assert sorted(p.name for p in tmp_path.glob("step_*")) == ["step_000050"]


def test_prune_ignores_incomplete_dirs_when_counting(tmp_path: Path) -> None:
    _save(tmp_path, 50)
    _save(tmp_path, 100)
    (tmp_path / "step_000100" / "meta.json").unlink()  # incomplete, not counted
    _save(tmp_path, 150, keep=2)
    assert (tmp_path / "step_000050").is_dir()
    assert (tmp_path / "step_000150").is_dir()


def test_rejects_keep_below_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep"):
        _save(tmp_path, 1, keep=0)


def test_resave_same_step_replaces(tmp_path: Path) -> None:
    _save(tmp_path, 50)
    path = _save(tmp_path, 50)
    assert load_checkpoint(path).train_state.step == 50
    assert not list(tmp_path.glob(".tmp-*"))


def test_restore_rng_reproduces_next_draws() -> None:
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    state = capture_rng()
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    restore_rng(state)
    again = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == again


def test_rng_survives_save_and_load(tmp_path: Path) -> None:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    path = _save(tmp_path, 10)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    restore_rng(load_checkpoint(path).rng)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


def test_restore_rng_rejects_missing_key() -> None:
    state = capture_rng()
    del state["numpy"]
    with pytest.raises(KeyError, match="numpy"):
        restore_rng(state)


def test_load_rejects_missing_meta(tmp_path: Path) -> None:
    path = tmp_path / "step_000001"
    path.mkdir()
    with pytest.raises(FileNotFoundError, match="step_000001"):
        load_checkpoint(path)


def test_load_rejects_unknown_format_version(tmp_path: Path) -> None:
    path = _save(tmp_path, 1)
    meta = json.loads((path / "meta.json").read_text())
    meta["format_version"] = 2
    (path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path)


_real_replace = os.replace
_real_rmtree = shutil.rmtree


def _replace_crashing_into(target: Path):
    def replace(src: str | Path, dst: str | Path) -> None:
        if Path(dst) == target:
            raise OSError("simulated crash")
        _real_replace(src, dst)

    return replace


def _partial_rmtree(path: str | Path) -> None:
    # worst-case partial delete: the model goes first, then the process dies
    model_file = Path(path) / "model.safetensors"
    if model_file.exists():
        model_file.unlink()
        raise OSError("simulated crash")
    _real_rmtree(path)


def test_crash_between_move_aside_and_swap_leaves_loadable_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save(tmp_path, 25)
    _save(tmp_path, 50)
    monkeypatch.setattr(os, "replace", _replace_crashing_into(tmp_path / "step_000050"))
    with pytest.raises(OSError, match="simulated crash"):
        _save(tmp_path, 50)
    monkeypatch.undo()

    found = latest_checkpoint(tmp_path)
    assert found == tmp_path / "step_000025"
    assert found is not None
    assert load_checkpoint(found).train_state.step == 25


def test_crash_while_deleting_old_copy_keeps_new_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save(tmp_path, 50)
    monkeypatch.setattr(shutil, "rmtree", _partial_rmtree)
    with pytest.raises(OSError, match="simulated crash"):
        _save(tmp_path, 50)
    monkeypatch.undo()

    found = latest_checkpoint(tmp_path)
    assert found == tmp_path / "step_000050"
    assert found is not None
    assert load_checkpoint(found).train_state.step == 50


def test_crash_while_pruning_never_leaves_half_deleted_complete_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save(tmp_path, 50)
    _save(tmp_path, 100)
    monkeypatch.setattr(shutil, "rmtree", _partial_rmtree)
    with pytest.raises(OSError, match="simulated crash"):
        _save(tmp_path, 150, keep=2)
    monkeypatch.undo()

    assert not (tmp_path / "step_000050").exists()  # moved aside, not left half-deleted
    assert latest_checkpoint(tmp_path) == tmp_path / "step_000150"
    assert load_checkpoint(tmp_path / "step_000100").train_state.step == 100


def test_reward_mode_round_trips(tmp_path: Path) -> None:
    path = save_checkpoint(
        tmp_path,
        train_state=TrainState(step=1, data_cursor=0, wandb_run_id=None, reward_mode="count_once"),
        model={},
        optimizers={},
        scaler=None,
        keep=1,
    )
    assert load_checkpoint(path).train_state.reward_mode == "count_once"


def test_checkpoint_without_reward_mode_is_rejected(tmp_path: Path) -> None:
    path = save_checkpoint(
        tmp_path,
        train_state=TrainState(step=1, data_cursor=0, wandb_run_id=None, reward_mode="normal"),
        model={},
        optimizers={},
        scaler=None,
        keep=1,
    )
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["train_state"]["reward_mode"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="reward_mode"):
        load_checkpoint(path)
