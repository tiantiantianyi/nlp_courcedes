"""XGrammar helpers for the shared Transformers VLM runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompiledXGrammar:
    grammar: Any
    tokenizer_vocab_size: int
    model_vocab_size: int
    stop_token_ids: tuple[int, ...]


def model_vocab_size(model_config: dict[str, Any]) -> int:
    candidates = [
        model_config.get("vocab_size"),
        model_config.get("text_config", {}).get("vocab_size"),
        model_config.get("llm_config", {}).get("vocab_size"),
    ]
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    raise ValueError("Could not determine the model LM-head vocabulary size")


def compile_json_schema(
    tokenizer: Any,
    schema: dict[str, Any],
    vocab_size: int,
    max_whitespace: int,
) -> CompiledXGrammar:
    import xgrammar as xgr

    tokenizer_vocab_size = max(tokenizer.get_vocab().values()) + 1
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=vocab_size,
    )
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
    return CompiledXGrammar(
        grammar=compiler.compile_json_schema(
            schema,
            strict_mode=True,
            max_whitespace_cnt=max_whitespace,
            any_order=False,
        ),
        tokenizer_vocab_size=tokenizer_vocab_size,
        model_vocab_size=vocab_size,
        stop_token_ids=tuple(tokenizer_info.stop_token_ids),
    )


def new_logits_processor(compiled_grammar: CompiledXGrammar) -> Any:
    from xgrammar.contrib.hf import LogitsProcessor

    class PaddedVocabLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            scores = super().__call__(input_ids, scores)
            scores[..., compiled_grammar.tokenizer_vocab_size :] = float("-inf")
            return scores

    return PaddedVocabLogitsProcessor(compiled_grammar.grammar)
