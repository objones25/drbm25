from dataclasses import replace
from pathlib import Path

import pytest

from drbm25.config import ConfigError, load_config, validate_for_runtime

CONFIGS = Path(__file__).parents[1] / "configs"
EXPANSE = CONFIGS / "expanse_v100_lora.yaml"
RUNPOD = CONFIGS / "runpod_a100_full.yaml"
SMOKE = CONFIGS / "smoke_cpu.yaml"


def test_loads_expanse_profile_values():
    cfg = load_config(EXPANSE)
    assert cfg.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.model.precision == "fp16"
    assert cfg.model.train_mode == "lora"
    assert cfg.lora is not None
    assert cfg.lora.rank == 32
    assert cfg.ppo.batch_size == 64
    assert cfg.ppo.kl_coef == pytest.approx(0.001)
    assert cfg.rollout.temperature == pytest.approx(0.6)
    assert cfg.retrieval.k1 == pytest.approx(0.9)
    assert cfg.retrieval.b == pytest.approx(0.4)
    assert cfg.run.total_steps == 1751
    assert isinstance(cfg.retrieval.index_dir, Path)


def test_loads_runpod_profile_values():
    cfg = load_config(RUNPOD)
    assert cfg.model.name == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.model.precision == "bf16"
    assert cfg.model.train_mode == "full"
    assert cfg.lora is None
    assert cfg.ppo.actor_lr == pytest.approx(1e-6)
    assert cfg.ppo.critic_lr == pytest.approx(1e-5)
    assert cfg.rollout.backend == "vllm"


def test_loads_smoke_profile_values():
    cfg = load_config(SMOKE)
    assert cfg.model.name == "tiny-local"
    assert cfg.model.precision == "fp32"
    assert cfg.run.total_steps == 4
    assert cfg.ppo.batch_size == 4
    assert cfg.ppo.mini_batch_size == 2
    assert cfg.ppo.micro_batch_size == 1
    assert cfg.rollout.max_new_tokens == 16
    assert cfg.rollout.backend == "hf"
    assert validate_for_runtime(cfg, device_capability=None, world_size=1) is None


def test_override_changes_nested_field():
    assert load_config(EXPANSE, {"run.total_steps": "20"}).run.total_steps == 20


def test_override_coerces_float_and_path():
    cfg = load_config(EXPANSE, {"ppo.kl_coef": "0.01", "run.output_dir": "/x/y"})
    assert cfg.ppo.kl_coef == pytest.approx(0.01)
    assert cfg.run.output_dir == Path("/x/y")


def test_rejects_unknown_key():
    with pytest.raises(ConfigError, match=r"unknown key 'ppo\.batchsize'"):
        load_config(EXPANSE, {"ppo.batchsize": "8"})


def test_rejects_bad_enum_value():
    with pytest.raises(ConfigError, match=r"model\.precision"):
        load_config(EXPANSE, {"model.precision": "fp8"})


def test_rejects_unparseable_override():
    with pytest.raises(ConfigError, match=r"ppo\.batch_size"):
        load_config(EXPANSE, {"ppo.batch_size": "many"})


def test_rejects_bf16_on_volta():
    cfg = load_config(RUNPOD)
    with pytest.raises(ConfigError, match=r"bf16 requires compute capability >= 8\.0"):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=2)


def test_accepts_bf16_on_ampere():
    cfg = load_config(RUNPOD)
    assert validate_for_runtime(cfg, device_capability=(8, 0), world_size=2) is None


def test_rejects_batch_not_divisible():
    cfg = load_config(EXPANSE)
    with pytest.raises(ConfigError, match="batch_size 64"):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=3)


def test_rejects_micro_batch_not_dividing_per_rank_mini_batch():
    cfg = load_config(EXPANSE, {"ppo.micro_batch_size": "3"})
    with pytest.raises(ConfigError, match="micro_batch_size 3"):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=1)


def test_rejects_batch_not_multiple_of_mini_batch():
    cfg = load_config(EXPANSE, {"ppo.mini_batch_size": "24"})
    with pytest.raises(ConfigError, match="mini_batch_size 24"):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=1)


def test_rejects_lora_mode_without_lora_section():
    cfg = replace(load_config(EXPANSE), lora=None)
    with pytest.raises(ConfigError, match="lora"):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=1)


def test_rejects_full_mode_with_lora_section():
    cfg = replace(load_config(RUNPOD), lora=load_config(EXPANSE).lora)
    with pytest.raises(ConfigError, match="lora"):
        validate_for_runtime(cfg, device_capability=(8, 0), world_size=1)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ppo.kl_coef", "-0.1"),
        ("ppo.actor_lr", "0"),
        ("ppo.critic_lr", "-1e-5"),
        ("ppo.clip_ratio", "1.0"),
        ("ppo.clip_ratio", "0"),
    ],
)
def test_rejects_reward_range_breaking_values(key, value):
    cfg = load_config(EXPANSE, {key: value})
    with pytest.raises(ConfigError, match=key.split(".")[1]):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=1)
