"""Transformers 5 compatibility for LM Format Enforcer 0.11.3.

LMFE 0.11.3 imports PreTrainedTokenizerBase from a Transformers 4 module path.
The upstream main branch already uses the public Transformers 5 import. This
module keeps the integration local until a released LMFE version includes it.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from lmformatenforcer.characterlevelparser import CharacterLevelParser
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData
from transformers import PreTrainedTokenizerBase


def _regular_tokens(
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
) -> list[tuple[int, str, bool]]:
    token_0 = tokenizer.encode("0")[-1]
    regular_tokens: list[tuple[int, str, bool]] = []
    for token_id in range(vocab_size):
        if token_id in tokenizer.all_special_ids:
            continue
        decoded_after_0 = tokenizer.decode([token_0, token_id])[1:]
        decoded_regular = tokenizer.decode([token_id])
        is_word_start = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_id, decoded_after_0, is_word_start))
    return regular_tokens


def _decode(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        clean_up_tokenization_spaces=False,
    ).rstrip("�")


def tokenizer_data(
    tokenizer: PreTrainedTokenizerBase,
) -> TokenEnforcerTokenizerData:
    vocab_size = len(tokenizer)
    return TokenEnforcerTokenizerData(
        _regular_tokens(tokenizer, vocab_size),
        functools.partial(_decode, tokenizer),
        tokenizer.eos_token_id,
        False,
        vocab_size,
    )


class PrefixAllowedTokens:
    def __init__(self, enforcer: TokenEnforcer) -> None:
        self.token_enforcer = enforcer

    def __call__(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        del batch_id
        return self.token_enforcer.get_allowed_tokens(
            input_ids.tolist()
        ).allowed_tokens


def build_prefix_allowed_tokens_fn(
    tokenizer: PreTrainedTokenizerBase,
    parser: CharacterLevelParser,
) -> Any:
    return PrefixAllowedTokens(TokenEnforcer(tokenizer_data(tokenizer), parser))
