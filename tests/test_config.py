from dataclasses import replace
from pathlib import Path

import pytest
import yaml

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


# --- non-positive ints and non-finite floats -------------------------------
@pytest.mark.parametrize(
    ("key", "value", "world_size"),
    [
        ("ppo.mini_batch_size", "0", 1),
        ("ppo.batch_size", "0", 1),
        ("ppo.micro_batch_size", "0", 1),
        ("ppo.mini_batch_size", "-16", 1),
    ],
)
def test_rejects_non_positive_batch_ints_as_config_error(key, value, world_size):
    cfg = load_config(EXPANSE, {key: value})
    with pytest.raises(ConfigError, match=key.split(".")[1]):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=world_size)


def test_rejects_zero_world_size():
    with pytest.raises(ConfigError, match="world_size"):
        validate_for_runtime(load_config(EXPANSE), device_capability=(7, 0), world_size=0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ppo.kl_coef", "nan"),
        ("ppo.kl_coef", "inf"),
        ("ppo.actor_lr", "nan"),
        ("ppo.actor_lr", "inf"),
        ("ppo.critic_lr", "inf"),
        ("ppo.clip_ratio", "nan"),
    ],
)
def test_rejects_non_finite_floats(key, value):
    cfg = load_config(EXPANSE, {key: value})
    with pytest.raises(ConfigError, match=key.split(".")[1]):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=1)


def test_rejects_empty_target_modules():
    cfg = load_config(EXPANSE, {"lora.target_modules": ""})
    with pytest.raises(ConfigError, match="target_modules"):
        validate_for_runtime(cfg, device_capability=(7, 0), world_size=1)


# --- strict typing of the YAML file itself ---------------------------------
def _write_variant(tmp_path, mutate):
    raw = yaml.safe_load(EXPANSE.read_text())
    mutate(raw)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_rejects_string_where_int_expected(tmp_path):
    p = _write_variant(tmp_path, lambda r: r["ppo"].update(batch_size="64"))
    with pytest.raises(ConfigError, match=r"ppo\.batch_size: expected int"):
        load_config(p)


def test_rejects_bool_where_int_expected(tmp_path):
    p = _write_variant(tmp_path, lambda r: r["ppo"].update(batch_size=True))
    with pytest.raises(ConfigError, match=r"ppo\.batch_size: expected int"):
        load_config(p)


def test_rejects_missing_key(tmp_path):
    p = _write_variant(tmp_path, lambda r: r["ppo"].pop("kl_coef"))
    with pytest.raises(ConfigError, match=r"missing key 'ppo\.kl_coef'"):
        load_config(p)


def test_rejects_unknown_key_in_yaml_file(tmp_path):
    p = _write_variant(tmp_path, lambda r: r["ppo"].update(batchsize=8))
    with pytest.raises(ConfigError, match=r"unknown key 'ppo\.batchsize'"):
        load_config(p)


def test_rejects_non_mapping_section(tmp_path):
    p = _write_variant(tmp_path, lambda r: r.update(ppo=[1, 2]))
    with pytest.raises(ConfigError, match=r"ppo: expected a mapping"):
        load_config(p)


def test_rejects_non_mapping_top_level(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_config(p)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "nope.yaml")


def test_rejects_bad_target_modules_type(tmp_path):
    p = _write_variant(tmp_path, lambda r: r["lora"].update(target_modules=[1, 2]))
    with pytest.raises(ConfigError, match=r"lora\.target_modules: expected a list of strings"):
        load_config(p)


# --- overrides: tuple, bool, null ------------------------------------------
def test_override_parses_tuple():
    cfg = load_config(EXPANSE, {"lora.target_modules": "q_proj,v_proj"})
    assert cfg.lora is not None
    assert cfg.lora.target_modules == ("q_proj", "v_proj")


def test_override_parses_bool():
    assert load_config(EXPANSE, {"model.gradient_checkpointing": "false"}).model.gradient_checkpointing is False
    assert load_config(EXPANSE, {"model.gradient_checkpointing": "true"}).model.gradient_checkpointing is True


def test_override_rejects_bad_bool():
    with pytest.raises(ConfigError, match=r"model\.gradient_checkpointing"):
        load_config(EXPANSE, {"model.gradient_checkpointing": "yes"})


def test_override_null_sets_optional_none():
    cfg = load_config(EXPANSE, {"run.wandb_entity": "someone"})
    assert cfg.run.wandb_entity == "someone"
    assert load_config(EXPANSE, {"run.wandb_entity": "null"}).run.wandb_entity is None


def test_override_rejects_section_key():
    with pytest.raises(ConfigError, match="is a section"):
        load_config(EXPANSE, {"ppo": "x"})


def test_override_rejects_key_below_leaf():
    with pytest.raises(ConfigError, match=r"unknown key 'ppo\.batch_size\.x'"):
        load_config(EXPANSE, {"ppo.batch_size.x": "1"})


@pytest.mark.parametrize("path", [EXPANSE, RUNPOD, SMOKE], ids=lambda p: p.stem)
def test_profile_pins_hf_revision_to_commit_sha(path: Path):
    assert load_config(path).data.hf_revision == "44eb79992e8c5ed4784dbe6559f8b097d7d03855"


def test_profiles_state_expected_index_size():
    assert load_config(EXPANSE).retrieval.expected_docs == 800_000
    assert load_config(RUNPOD).retrieval.expected_docs == 800_000
    # The smoke profile runs on the 1,002-doc toy index built by the test suite.
    assert load_config(SMOKE).retrieval.expected_docs == 1002


def test_smoke_profile_uses_lora_layout():
    cfg = load_config(SMOKE)
    assert cfg.model.train_mode == "lora"
    assert cfg.lora is not None
    assert cfg.lora.dropout == pytest.approx(0.0)


@pytest.mark.parametrize("path", [EXPANSE, RUNPOD, SMOKE], ids=lambda p: p.stem)
def test_profiles_record_split_seed(path: Path):
    assert load_config(path).data.split_seed == 1234


@pytest.mark.parametrize("path", [EXPANSE, RUNPOD], ids=lambda p: p.stem)
def test_real_profiles_keep_one_checkpoint_and_validate_on_500_train_queries(path: Path):
    cfg = load_config(path)
    assert cfg.run.keep_ckpts == 1
    assert cfg.data.val_size == 500


def test_runpod_profile_uses_micro_batch_8():
    cfg = load_config(RUNPOD)
    assert cfg.ppo.micro_batch_size == 8
    assert (cfg.ppo.batch_size, cfg.ppo.mini_batch_size) == (64, 16)
    assert validate_for_runtime(cfg, device_capability=(8, 0), world_size=2) is None


def test_split_seed_is_required(tmp_path: Path):
    p = _write_variant(tmp_path, lambda r: r["data"].pop("split_seed", None))
    with pytest.raises(ConfigError, match=r"missing key 'data\.split_seed'"):
        load_config(p)


def test_rejects_non_positive_expected_docs():
    cfg = load_config(SMOKE, {"retrieval.expected_docs": "0"})
    with pytest.raises(ConfigError, match="expected_docs"):
        validate_for_runtime(cfg, device_capability=None, world_size=1)
