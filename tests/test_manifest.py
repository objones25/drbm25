"""run_manifest.json: every seed, the resolved config and its hash, code and environment versions."""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from drbm25.config import load_config
from drbm25.manifest import (
    Environment,
    build_manifest,
    config_sha256,
    read_manifest,
    seeds,
    update_manifest,
    write_manifest,
)

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"
RUNPOD = REPO / "configs" / "runpod_a100_full.yaml"
START = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
ENV = Environment(
    git_sha="f" * 40,
    git_dirty=False,
    versions={"python": "3.12.0", "torch": "2.10.0", "transformers": "4.57.6", "peft": "0.21.2", "vllm": None},
    cuda=None,
    driver=None,
    gpus=(),
    hostname="host-a",
)


def _manifest(overrides: dict[str, str] | None = None, *, world_size: int = 2, path: Path = SMOKE) -> dict:
    cfg = load_config(path, overrides or {})
    return build_manifest(
        cfg,
        run_name="normal",
        reward_mode="normal",
        world_size=world_size,
        index_checksum="c" * 64,
        env=ENV,
        started_utc=START,
    )


def test_seeds_record_every_seed_and_its_derivation() -> None:
    cfg = load_config(RUNPOD, {"run.seed": "7", "data.split_seed": "99"})
    got = seeds(cfg, world_size=2)
    assert got == {
        "run_seed": 7,
        "split_seed": 99,
        "data_order_seed": 7,
        "model_init_seed": 7,
        "sampling_seeds": [7, 8],
        "vllm_seeds": [7, 8],
    }


def test_seeds_have_no_vllm_seed_on_the_hf_backend() -> None:
    assert seeds(load_config(SMOKE), world_size=1)["vllm_seeds"] is None


def test_manifest_holds_seeds_config_hash_versions_and_environment() -> None:
    cfg = load_config(SMOKE)
    manifest = _manifest()
    assert manifest["format_version"] == 1
    assert manifest["seeds"] == seeds(cfg, world_size=2)
    assert set(manifest["seed_derivation"]) == {*manifest["seeds"], "resume_reseed", "bootstrap_seed"}
    assert manifest["config"]["run"]["seed"] == 1234
    assert manifest["config"]["data"]["split_seed"] == 1234
    assert manifest["config_sha256"] == config_sha256(cfg)
    assert manifest["effective_micro_batch_size"] == cfg.ppo.micro_batch_size
    assert manifest["git"] == {"sha": "f" * 40, "dirty": False}
    assert manifest["dataset"] == {"hf_dataset": cfg.data.hf_dataset, "hf_revision": cfg.data.hf_revision}
    assert manifest["model"] == {"name": cfg.model.name, "revision": cfg.model.revision}
    assert manifest["index"] == {"checksum": "c" * 64, "n_docs": cfg.retrieval.expected_docs}
    assert manifest["versions"]["torch"] == "2.10.0"
    assert manifest["hardware"] == {"cuda": None, "driver": None, "gpus": []}
    assert manifest["world_size"] == 2
    assert manifest["hostname"] == "host-a"
    assert manifest["start_time_utc"] == "2026-11-20T12:00:00+00:00"
    assert manifest["resumed_at"] == []
    json.dumps(manifest)  # plain JSON throughout


def test_config_sha256_is_the_hash_of_the_canonical_resolved_config() -> None:
    cfg = load_config(SMOKE)
    canonical = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), default=str)
    assert config_sha256(cfg) == hashlib.sha256(canonical.encode()).hexdigest()
    assert config_sha256(load_config(SMOKE, {"ppo.micro_batch_size": "2"})) != config_sha256(cfg)


def test_effective_micro_batch_size_follows_the_override() -> None:
    assert _manifest({"ppo.micro_batch_size": "4"}, path=RUNPOD)["effective_micro_batch_size"] == 4


def test_collect_environment_lists_the_pinned_library_versions() -> None:
    from drbm25.manifest import collect_environment

    env = collect_environment()
    assert {"python", "torch", "transformers", "peft", "vllm", "pyserini", "wandb"} <= env.versions.keys()
    assert env.versions["torch"] is not None
    assert env.versions["torch"].startswith("2.10.0")
    assert env.hostname


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    manifest = _manifest()
    write_manifest(tmp_path, manifest)
    assert read_manifest(tmp_path) == manifest
    assert list(tmp_path.glob(".tmp-*")) == []
    assert read_manifest(tmp_path / "missing") is None


def test_first_start_returns_the_new_manifest_unchanged() -> None:
    assert update_manifest(None, _manifest(), step=0) == _manifest()


def test_resume_appends_resumed_at_and_keeps_seeds_and_start() -> None:
    first = _manifest()
    second = _manifest({"ppo.micro_batch_size": "1"})
    second["hostname"] = "host-b"
    merged = update_manifest(first, second, step=50)
    assert merged["seeds"] == first["seeds"]
    assert merged["start_time_utc"] == first["start_time_utc"]
    assert merged["config_sha256"] == first["config_sha256"]
    assert merged["resumed_at"] == [
        {
            "time_utc": second["start_time_utc"],
            "step": 50,
            "config_sha256": second["config_sha256"],
            "git": second["git"],
            "hostname": "host-b",
            "world_size": 2,
            "effective_micro_batch_size": second["effective_micro_batch_size"],
        }
    ]
    again = update_manifest(merged, second, step=100)
    assert [r["step"] for r in again["resumed_at"]] == [50, 100]
    assert first["resumed_at"] == []  # inputs are not mutated


@pytest.mark.parametrize(("key", "value"), [("run.seed", "99"), ("data.split_seed", "99")])
def test_resume_refuses_a_changed_seed(key: str, value: str) -> None:
    with pytest.raises(ValueError, match="seed"):
        update_manifest(_manifest(), _manifest({key: value}), step=50)
