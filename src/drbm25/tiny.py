"""Tiny randomly initialised models for tests and CPU smoke runs (no download)."""

import torch
from transformers import PreTrainedModel, Qwen2Config, Qwen2ForCausalLM


def build_tiny_qwen2(seed: int = 0) -> PreTrainedModel:
    """A 2-layer fp32 Qwen2 causal LM on CPU, initialised from ``seed``.

    The global torch RNG is left untouched (``fork_rng``), so building a model inside a
    test does not shift the random stream of whatever runs next.
    """
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen2ForCausalLM(config)
    return model
