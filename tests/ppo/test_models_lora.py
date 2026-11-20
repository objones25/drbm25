"""LoRA actor/critic on one frozen base: logprobs, adapter isolation, state, merge, checkpointing."""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, Qwen2ForCausalLM

from drbm25.config import LoraCfg
from drbm25.ppo.models import AdapterSwitchError, LoraPolicyModels, PolicyModels, token_logprobs
from drbm25.tiny import build_tiny_qwen2

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA = LoraCfg(rank=8, alpha=16, dropout=0.0, target_modules=TARGETS)
B, L = 2, 6


def _policy(base: PreTrainedModel, *, gradient_checkpointing: bool = False) -> LoraPolicyModels:
    return LoraPolicyModels(base, LORA, gradient_checkpointing=gradient_checkpointing)


def _batch(seed: int = 1) -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 128, (B, L), generator=gen)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[1, :2] = 0  # left padding on the second row
    return ids, mask


def _actor_loss(policy: PolicyModels, ids: Tensor, mask: Tensor) -> Tensor:
    return -token_logprobs(policy.actor_logits(ids, mask), ids).mean()


def _critic_loss(policy: PolicyModels, ids: Tensor, mask: Tensor) -> Tensor:
    return ((policy.critic_values(ids, mask) - 1.0) ** 2).mean()


def _step(params: list[nn.Parameter], loss_fn: Callable[[], Tensor], *, steps: int = 1, lr: float = 0.5) -> None:
    opt = torch.optim.SGD(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()


def _zero_grads(params: list[nn.Parameter]) -> None:
    for p in params:
        p.grad = None


def _snapshot(policy: LoraPolicyModels, base: PreTrainedModel) -> dict[str, dict[str, Tensor]]:
    actor = {id(p) for p in policy.actor_parameters()}
    critic = {id(p) for p in policy.critic_parameters()}
    groups: dict[str, dict[str, Tensor]] = {"actor": {}, "critic": {}, "base": {}}
    for name, p in base.named_parameters():
        group = "actor" if id(p) in actor else "critic" if id(p) in critic else "base"
        groups[group][name] = p.detach().clone()
    in_base = {id(p) for p in base.parameters()}
    head = [p for p in policy.critic_parameters() if id(p) not in in_base]
    assert len(head) == 2  # value head weight and bias live outside the base model
    for i, p in enumerate(head):
        groups["critic"][f"value_head.{i}"] = p.detach().clone()
    return groups


def _changed(before: dict[str, Tensor], after: dict[str, Tensor]) -> set[str]:
    assert before.keys() == after.keys()
    return {k for k in before if not torch.equal(before[k], after[k])}


# --------------------------------------------------------------------------- token_logprobs
def test_token_logprobs_matches_manual_gather() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 7, generator=gen, dtype=torch.float64)
    ids = torch.randint(0, 7, (2, 5), generator=gen)
    expected = torch.empty(2, 4, dtype=torch.float64)
    for b in range(2):
        for t in range(4):
            expected[b, t] = torch.log_softmax(logits[b, t], dim=-1)[ids[b, t + 1]]
    out = token_logprobs(logits, ids)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, expected.float())


def test_token_logprobs_upcasts_half_logits_to_fp32() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(2, 5, 50, generator=gen) * 8).half()
    ids = torch.randint(0, 50, (2, 5), generator=gen)
    out = token_logprobs(logits, ids)
    assert out.dtype == torch.float32
    reference = torch.log_softmax(logits.double(), dim=-1)[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(out, reference.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("logits_shape", "ids_shape"),
    [((2, 5, 7), (2, 4)), ((2, 5, 7), (3, 5)), ((2, 5), (2, 5)), ((2, 1, 7), (2, 1))],
)
def test_token_logprobs_rejects_bad_shapes(logits_shape: tuple[int, ...], ids_shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="token_logprobs"):
        token_logprobs(torch.zeros(logits_shape), torch.zeros(ids_shape, dtype=torch.long))


# --------------------------------------------------------------------------- layout
def test_lora_policy_satisfies_protocol(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    assert isinstance(policy, PolicyModels)
    assert len(policy.decoder_layers()) == 2


def test_actor_and_critic_parameters_are_disjoint_lora_and_head(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    actor_ids = {id(p) for p in policy.actor_parameters()}
    critic_ids = {id(p) for p in policy.critic_parameters()}
    # 2 layers x 7 target modules x (A, B); critic adds the value head weight and bias.
    assert len(actor_ids) == 28
    assert len(critic_ids) == 30
    assert not actor_ids & critic_ids
    names = {id(p): n for n, p in tiny_qwen2.named_parameters()}
    assert all(".actor." in names[i] for i in actor_ids)
    assert all(p.requires_grad for p in policy.actor_parameters() + policy.critic_parameters())
    trainable_base = [n for n, p in tiny_qwen2.named_parameters() if p.requires_grad and "lora_" not in n]
    assert trainable_base == []


def test_actor_equals_ref_at_init(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    ids, mask = _batch()
    torch.testing.assert_close(policy.actor_logits(ids, mask), policy.ref_logits(ids, mask))


def test_critic_values_shape_and_zero_at_init(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    ids, mask = _batch()
    values = policy.critic_values(ids, mask)
    assert values.shape == (B, L)
    assert values.dtype == torch.float32
    torch.testing.assert_close(values, torch.zeros(B, L))


def test_ref_logits_equal_base_logits_at_init_and_after_actor_update(tiny_qwen2: PreTrainedModel) -> None:
    plain = build_tiny_qwen2(seed=0)
    ids, mask = _batch()
    with torch.no_grad():
        base_logits = plain(input_ids=ids, attention_mask=mask).logits
    policy = _policy(tiny_qwen2)
    ref0 = policy.ref_logits(ids, mask)
    assert not ref0.requires_grad
    torch.testing.assert_close(ref0, base_logits)

    _step(policy.actor_parameters(), lambda: _actor_loss(policy, ids, mask))

    assert not torch.allclose(policy.actor_logits(ids, mask), base_logits)
    torch.testing.assert_close(policy.ref_logits(ids, mask), base_logits)


def test_actor_step_changes_only_actor_params(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    ids, mask = _batch()
    before = _snapshot(policy, tiny_qwen2)
    _step(policy.actor_parameters(), lambda: _actor_loss(policy, ids, mask))
    after = _snapshot(policy, tiny_qwen2)
    assert _changed(before["base"], after["base"]) == set()
    assert _changed(before["critic"], after["critic"]) == set()
    lora_b = {k for k in before["actor"] if ".lora_B." in k}
    assert len(lora_b) == 14
    assert lora_b <= _changed(before["actor"], after["actor"])


def test_critic_step_changes_only_critic_params_and_value_head(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    ids, mask = _batch()
    before = _snapshot(policy, tiny_qwen2)
    # Two steps: the zero-initialised head blocks the gradient into the critic LoRA on step one.
    _step(policy.critic_parameters(), lambda: _critic_loss(policy, ids, mask), steps=2)
    after = _snapshot(policy, tiny_qwen2)
    assert _changed(before["base"], after["base"]) == set()
    assert _changed(before["actor"], after["actor"]) == set()
    changed = _changed(before["critic"], after["critic"])
    assert {"value_head.0", "value_head.1"} <= changed
    assert {k for k in before["critic"] if ".lora_B." in k} <= changed


def test_adapter_switching_does_not_leak_between_calls(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    ids, mask = _batch()
    _step(policy.actor_parameters(), lambda: _actor_loss(policy, ids, mask))
    _step(policy.critic_parameters(), lambda: _critic_loss(policy, ids, mask), steps=2)
    with torch.no_grad():
        actor = policy.actor_logits(ids, mask)
        values = policy.critic_values(ids, mask)
        gen_logits = policy.generation_module()(input_ids=ids, attention_mask=mask).logits
        assert isinstance(gen_logits, Tensor)
        torch.testing.assert_close(gen_logits, actor)
        torch.testing.assert_close(policy.critic_values(ids, mask), values)
        torch.testing.assert_close(policy.actor_logits(ids, mask), actor)


def test_critic_and_actor_losses_backprop_together(tiny_qwen2: PreTrainedModel) -> None:
    """Both forwards in one graph (switching adapter in between) give each side its own gradients."""
    policy = _policy(tiny_qwen2)
    ids, mask = _batch()
    _step(policy.critic_parameters(), lambda: _critic_loss(policy, ids, mask))  # make the head non-zero
    _zero_grads(policy.actor_parameters() + policy.critic_parameters())
    (_critic_loss(policy, ids, mask) + _actor_loss(policy, ids, mask)).backward()
    lora_b = [
        p for p in policy.actor_parameters() + policy.critic_parameters() if p.ndim == 2 and p.shape[1] == LORA.rank
    ]
    assert len(lora_b) == 28
    assert all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in lora_b)


# --------------------------------------------------------------------------- state
def test_trainable_state_roundtrip(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    ids, mask = _batch()
    _step(policy.actor_parameters(), lambda: _actor_loss(policy, ids, mask))
    _step(policy.critic_parameters(), lambda: _critic_loss(policy, ids, mask), steps=2)
    state = policy.trainable_state_dict()
    assert len(state) == 58

    fresh = _policy(build_tiny_qwen2(seed=0))
    fresh.load_trainable_state_dict(state)

    with torch.no_grad():
        torch.testing.assert_close(fresh.actor_logits(ids, mask), policy.actor_logits(ids, mask), rtol=0, atol=0)
        torch.testing.assert_close(fresh.critic_values(ids, mask), policy.critic_values(ids, mask), rtol=0, atol=0)
        torch.testing.assert_close(fresh.ref_logits(ids, mask), policy.ref_logits(ids, mask), rtol=0, atol=0)


def test_trainable_state_dict_is_a_copy(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    state = policy.trainable_state_dict()
    for tensor in state.values():
        tensor.add_(1.0)
    ids, mask = _batch()
    torch.testing.assert_close(policy.critic_values(ids, mask), torch.zeros(B, L))


def test_load_trainable_state_dict_rejects_missing_and_unexpected_keys(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    state = policy.trainable_state_dict()
    missing = dict(state)
    missing.pop("value_head.bias")
    with pytest.raises(KeyError, match=r"missing.*value_head\.bias"):
        policy.load_trainable_state_dict(missing)
    with pytest.raises(KeyError, match=r"unexpected.*bogus"):
        policy.load_trainable_state_dict({**state, "bogus": torch.zeros(1)})


def test_load_trainable_state_dict_rejects_shape_mismatch(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    state = policy.trainable_state_dict()
    with pytest.raises(ValueError, match=r"value_head\.weight"):
        policy.load_trainable_state_dict({**state, "value_head.weight": torch.zeros(2, 32)})


# --------------------------------------------------------------------------- merged export
def test_merged_actor_state_dict_reproduces_actor_logits(tiny_qwen2: PreTrainedModel) -> None:
    policy = _policy(tiny_qwen2)
    ids, mask = _batch()
    _step(policy.actor_parameters(), lambda: _actor_loss(policy, ids, mask), steps=3)
    with torch.no_grad():
        actor_before = policy.actor_logits(ids, mask)

    merged = policy.merged_actor_state_dict()

    plain = build_tiny_qwen2(seed=123)  # different init: every weight must come from `merged`
    assert set(merged) == set(plain.state_dict())
    plain.load_state_dict(merged, strict=True)
    with torch.no_grad():
        plain_logits = plain(input_ids=ids, attention_mask=mask).logits
        torch.testing.assert_close(plain_logits, actor_before)
        # the live model is untouched: same actor logits, and ref still equals the base
        torch.testing.assert_close(policy.actor_logits(ids, mask), actor_before, rtol=0, atol=0)
        torch.testing.assert_close(
            policy.ref_logits(ids, mask), build_tiny_qwen2(seed=0)(ids, attention_mask=mask).logits
        )


def test_merged_actor_state_dict_at_init_is_the_base(tiny_qwen2: PreTrainedModel) -> None:
    reference = build_tiny_qwen2(seed=0).state_dict()
    merged = _policy(tiny_qwen2).merged_actor_state_dict()
    assert set(merged) == set(reference)
    for key, tensor in reference.items():
        torch.testing.assert_close(merged[key], tensor, rtol=0, atol=0)


# --------------------------------------------------------------------------- precision
def test_adapters_and_value_head_stay_fp32_over_fp16_base() -> None:
    base = build_tiny_qwen2(seed=0).half()
    policy = _policy(base)
    assert {p.dtype for p in policy.actor_parameters() + policy.critic_parameters()} == {torch.float32}
    assert {p.dtype for n, p in base.named_parameters() if "lora_" not in n} == {torch.float16}
    ids, mask = _batch()
    assert policy.critic_values(ids, mask).dtype == torch.float32
    assert token_logprobs(policy.actor_logits(ids, mask), ids).dtype == torch.float32


# --------------------------------------------------------------------------- checkpointing
def _count_forwards(module: nn.Module) -> list[int]:
    calls = [0]

    def hook(_module: nn.Module, _args: object, _output: object) -> None:
        calls[0] += 1

    module.register_forward_hook(hook)
    return calls


def _actor_grads_and_layer0_forwards(policy: LoraPolicyModels, ids: Tensor, mask: Tensor) -> tuple[list[Tensor], int]:
    calls = _count_forwards(policy.decoder_layers()[0])
    _zero_grads(policy.actor_parameters())
    _actor_loss(policy, ids, mask).backward()
    grads = [p.grad for p in policy.actor_parameters()]
    assert all(g is not None for g in grads)
    return [g for g in grads if g is not None], calls[0]


def test_gradient_checkpointing_grads_match_no_checkpointing() -> None:
    ids, mask = _batch()
    plain = _policy(build_tiny_qwen2(seed=0))
    _step(plain.actor_parameters(), lambda: _actor_loss(plain, ids, mask))  # non-zero B so A gets grads
    checkpointed = _policy(build_tiny_qwen2(seed=0), gradient_checkpointing=True)
    checkpointed.load_trainable_state_dict(plain.trainable_state_dict())

    expected, plain_calls = _actor_grads_and_layer0_forwards(plain, ids, mask)
    got, checkpointed_calls = _actor_grads_and_layer0_forwards(checkpointed, ids, mask)

    assert (plain_calls, checkpointed_calls) == (1, 2)  # the checkpointed layer re-runs during backward
    assert all(bool(g.abs().sum() > 0) for g in got)
    torch.testing.assert_close(got, expected)


def test_gradient_checkpointing_ref_and_critic_match_no_checkpointing() -> None:
    ids, mask = _batch()
    plain = _policy(build_tiny_qwen2(seed=0))
    _step(plain.critic_parameters(), lambda: _critic_loss(plain, ids, mask), steps=2)
    checkpointed = _policy(build_tiny_qwen2(seed=0), gradient_checkpointing=True)
    checkpointed.load_trainable_state_dict(plain.trainable_state_dict())
    torch.testing.assert_close(checkpointed.ref_logits(ids, mask), plain.ref_logits(ids, mask))
    torch.testing.assert_close(checkpointed.critic_values(ids, mask), plain.critic_values(ids, mask))
    with torch.no_grad():  # rollout-time forwards: no checkpoint warning, same numbers
        torch.testing.assert_close(checkpointed.actor_logits(ids, mask), plain.actor_logits(ids, mask))
        torch.testing.assert_close(checkpointed.critic_values(ids, mask), plain.critic_values(ids, mask))
    _zero_grads(checkpointed.critic_parameters())
    _critic_loss(checkpointed, ids, mask).backward()
    assert all(p.grad is not None for p in checkpointed.critic_parameters())


# --------------------------------------------------------------------------- checkpointing + adapter switch
def _random_lora_pair() -> tuple[LoraPolicyModels, LoraPolicyModels]:
    """Same random (non-zero) adapters and head on a plain and a checkpointed policy."""
    plain = _policy(build_tiny_qwen2(seed=0))
    gen = torch.Generator().manual_seed(0)
    state = {k: torch.randn(v.shape, generator=gen) * 0.1 for k, v in plain.trainable_state_dict().items()}
    plain.load_trainable_state_dict(state)
    checkpointed = _policy(build_tiny_qwen2(seed=0), gradient_checkpointing=True)
    checkpointed.load_trainable_state_dict(state)
    return plain, checkpointed


def _grads(params: list[nn.Parameter]) -> list[Tensor]:
    grads = [p.grad for p in params]
    assert all(g is not None for g in grads)
    return [g.clone() for g in grads if g is not None]


def _sequential_grads(policy: LoraPolicyModels, ids: Tensor, mask: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    _zero_grads(policy.actor_parameters() + policy.critic_parameters())
    _actor_loss(policy, ids, mask).backward()
    _critic_loss(policy, ids, mask).backward()
    return _grads(policy.actor_parameters()), _grads(policy.critic_parameters())


def test_checkpointing_rejects_switching_adapter_before_backward() -> None:
    _, checkpointed = _random_lora_pair()
    ids, mask = _batch()
    actor_loss = _actor_loss(checkpointed, ids, mask)
    with pytest.raises(AdapterSwitchError, match="backward the actor loss before running critic"):
        checkpointed.critic_values(ids, mask)
    actor_loss.backward()  # the pending actor output is still correct after the refused switch
    _critic_loss(checkpointed, ids, mask).backward()


def test_checkpointing_sequential_backward_matches_plain_grads() -> None:
    plain, checkpointed = _random_lora_pair()
    ids, mask = _batch()
    expected_actor, expected_critic = _sequential_grads(plain, ids, mask)
    got_actor, got_critic = _sequential_grads(checkpointed, ids, mask)
    torch.testing.assert_close(got_actor, expected_actor)
    torch.testing.assert_close(got_critic, expected_critic)


def _actor_grads_with_detour(policy: LoraPolicyModels, ids: Tensor, mask: Tensor) -> list[Tensor]:
    """Actor forward, then no-grad ref and critic forwards, then the actor backward."""
    _zero_grads(policy.actor_parameters())
    loss = _actor_loss(policy, ids, mask)
    policy.ref_logits(ids, mask)
    with torch.no_grad():
        policy.critic_values(ids, mask)
    loss.backward()
    return _grads(policy.actor_parameters())


def test_checkpointing_no_grad_forwards_between_forward_and_backward_keep_grads_correct() -> None:
    plain, checkpointed = _random_lora_pair()
    ids, mask = _batch()
    torch.testing.assert_close(
        _actor_grads_with_detour(checkpointed, ids, mask), _actor_grads_with_detour(plain, ids, mask)
    )


def test_discard_pending_allows_switching_after_an_abandoned_forward() -> None:
    _, checkpointed = _random_lora_pair()
    ids, mask = _batch()
    checkpointed.actor_logits(ids, mask)  # output intentionally never backpropagated
    checkpointed.discard_pending()
    _critic_loss(checkpointed, ids, mask).backward()
    assert all(p.grad is not None for p in checkpointed.critic_parameters())


def test_generation_module_refused_while_a_backward_is_pending() -> None:
    _, checkpointed = _random_lora_pair()
    ids, mask = _batch()
    loss = _actor_loss(checkpointed, ids, mask)
    with pytest.raises(AdapterSwitchError, match="generation"):
        checkpointed.generation_module()
    loss.backward()
    assert isinstance(checkpointed.generation_module(), nn.Module)


def test_without_checkpointing_switching_before_backward_is_allowed() -> None:
    plain, _ = _random_lora_pair()
    ids, mask = _batch()
    _zero_grads(plain.actor_parameters() + plain.critic_parameters())
    (_actor_loss(plain, ids, mask) + _critic_loss(plain, ids, mask)).backward()
    assert len(_grads(plain.actor_parameters())) == 28
    assert len(_grads(plain.critic_parameters())) == 30


def _joint_grads(policy: LoraPolicyModels, ids: Tensor, mask: Tensor, *, ref_between: bool) -> list[Tensor]:
    _zero_grads(policy.actor_parameters() + policy.critic_parameters())
    actor_loss = _actor_loss(policy, ids, mask)
    critic_loss = _critic_loss(policy, ids, mask)
    if ref_between:
        policy.ref_logits(ids, mask)
    (actor_loss + critic_loss).backward()
    return _grads(policy.actor_parameters()) + _grads(policy.critic_parameters())


def test_ref_logits_between_forward_and_backward_keeps_joint_grads() -> None:
    plain, _ = _random_lora_pair()
    ids, mask = _batch()
    expected = _joint_grads(plain, ids, mask, ref_between=False)
    torch.testing.assert_close(_joint_grads(plain, ids, mask, ref_between=True), expected)


def _actor_grads_after_detours(policy: LoraPolicyModels, ids: Tensor, mask: Tensor, *, detour: bool) -> list[Tensor]:
    _zero_grads(policy.actor_parameters())
    loss = _actor_loss(policy, ids, mask)
    if detour:
        with torch.no_grad():
            policy.critic_values(ids, mask)
        policy.ref_logits(ids, mask)
    loss.backward()
    return _grads(policy.actor_parameters())


def test_no_grad_critic_then_ref_between_forward_and_backward_keeps_actor_grads() -> None:
    plain, _ = _random_lora_pair()
    ids, mask = _batch()
    expected = _actor_grads_after_detours(plain, ids, mask, detour=False)
    torch.testing.assert_close(_actor_grads_after_detours(plain, ids, mask, detour=True), expected)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_ref_logits_leaves_every_lora_param_trainable(gradient_checkpointing: bool) -> None:
    policy = _policy(build_tiny_qwen2(seed=0), gradient_checkpointing=gradient_checkpointing)
    ids, mask = _batch()
    with torch.no_grad():
        policy.critic_values(ids, mask)
    policy.ref_logits(ids, mask)
    assert all(p.requires_grad for p in policy.actor_parameters() + policy.critic_parameters())


# --------------------------------------------------------------------------- overfit
def test_lora_overfits_one_batch(tiny_qwen2: PreTrainedModel) -> None:
    """pytorch-skill overfit protocol: the actor LoRA alone can memorise one fixed batch.

    The fixture's tied embedding (std 0.02) and final RMSNorm are frozen, which bounds
    |logit| <= ||h|| * ||e|| ~ 0.84 and so CE >= 3.2 whatever the LoRA learns. Scaling the
    frozen embedding x20 gives a realistic logit range without training it. Rank 32 /
    alpha 64 and grad clip 1.0 are the project's LoRA and PPO settings.
    """
    embedding = tiny_qwen2.get_input_embeddings().weight
    assert isinstance(embedding, Tensor)
    with torch.no_grad():
        embedding.mul_(20.0)
    policy = LoraPolicyModels(
        tiny_qwen2, LoraCfg(rank=32, alpha=64, dropout=0.0, target_modules=TARGETS), gradient_checkpointing=False
    )
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 128, (4, 12), generator=gen)
    mask = torch.ones_like(ids)
    params = policy.actor_parameters()
    opt = torch.optim.SGD(params, lr=1.0)
    losses: list[float] = []
    for _ in range(100):
        opt.zero_grad()
        loss = _actor_loss(policy, ids, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 4.0
    assert losses[-1] < 0.5


# --------------------------------------------------------------------------- tiny model
def test_build_tiny_qwen2_is_seeded_and_leaves_global_rng_alone() -> None:
    state = torch.random.get_rng_state()
    a, b, c = build_tiny_qwen2(seed=0), build_tiny_qwen2(seed=0), build_tiny_qwen2(seed=1)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert isinstance(a, Qwen2ForCausalLM)
    wa, wb, wc = (m.get_input_embeddings().weight for m in (a, b, c))
    assert isinstance(wa, Tensor)
    assert isinstance(wb, Tensor)
    assert isinstance(wc, Tensor)
    assert torch.equal(wa, wb)
    assert not torch.equal(wa, wc)
    assert a.config.vocab_size == 128
    assert a.config.hidden_size == 32
    assert a.config.intermediate_size == 64
    assert a.config.num_hidden_layers == 2
    assert a.config.num_attention_heads == 4
    assert a.config.num_key_value_heads == 2
    assert a.config.tie_word_embeddings
    assert a.config._attn_implementation == "sdpa"
    assert {p.dtype for p in a.parameters()} == {torch.float32}
