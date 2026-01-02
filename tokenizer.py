from __future__ import annotations

from typing import Protocol

import tiktoken


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


def get_gpt2_tokenizer() -> Tokenizer:
    """Return the standard GPT-2 BPE tokenizer (vocab size 50257)."""
    return tiktoken.get_encoding("gpt2")


