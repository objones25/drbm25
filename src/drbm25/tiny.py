"""Tiny randomly initialised models for tests and CPU smoke runs (no download)."""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedModel, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

_TINY_VOCAB_SIZE = 128
_PAD, _EOS, _UNK = "<pad>", "<eos>", "<unk>"


def build_tiny_qwen2(seed: int = 0) -> PreTrainedModel:
    """A 2-layer fp32 Qwen2 causal LM on CPU, initialised from ``seed``.

    The global torch RNG is left untouched (``fork_rng``), so building a model inside a
    test does not shift the random stream of whatever runs next.
    """
    config = Qwen2Config(
        vocab_size=_TINY_VOCAB_SIZE,
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


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer with exactly 128 ids, matching ``build_tiny_qwen2``'s vocab.

    Ids 0-2 are ``<pad>``, ``<eos>``, ``<unk>``; ids 3-127 are the words ``w0`` .. ``w124``
    (whitespace-split). Pad and eos are distinct and padding is on the left.
    """
    words = [f"w{i}" for i in range(_TINY_VOCAB_SIZE - 3)]
    vocab = {tok: i for i, tok in enumerate([_PAD, _EOS, _UNK, *words])}
    core = Tokenizer(models.WordLevel(vocab, unk_token=_UNK))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # pyright: ignore[reportAttributeAccessIssue]
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token=_PAD,
        eos_token=_EOS,
        unk_token=_UNK,
        padding_side="left",
    )
