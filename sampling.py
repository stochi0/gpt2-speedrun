from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from model import GPT2Model


@dataclass
class SamplingConfig:
    strategy: Literal["greedy", "multinomial", "top_k", "top_p"] = "multinomial"
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    v, _ = torch.topk(logits, k, dim=-1)
    cutoff = v[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # remove tokens where cumulative prob exceeds p (keep at least 1 token)
    remove = cum > p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # scatter back to original order
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """
    logits: (B, V)
    returns: next_token (B, 1)
    """
    if cfg.temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / cfg.temperature

    if cfg.strategy == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    if cfg.strategy == "top_k":
        logits = _apply_top_k(logits, cfg.top_k)
    elif cfg.strategy == "top_p":
        logits = _apply_top_p(logits, cfg.top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model: GPT2Model,
    x: torch.Tensor,
    *,
    max_new_tokens: int,
    cfg: SamplingConfig | None = None,
) -> torch.Tensor:
    """
    Autoregressive generation loop.
    x: (B, T) token ids
    returns: (B, T + max_new_tokens)
    """
    if cfg is None:
        cfg = SamplingConfig()

    for _ in range(max_new_tokens):
        # crop to context window
        x_cond = x[:, -model.config.sequence_window_size :]
        logits = model(x_cond)[:, -1, :]  # (B, V)
        next_token = sample_next_token(logits, cfg)  # (B, 1)
        x = torch.cat([x, next_token], dim=1)
    return x
