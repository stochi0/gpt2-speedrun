from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken

# Sampling utilities live in `sampling.py` to keep this file focused on model architecture.
from sampling import SamplingConfig, generate

# gpt2 config

torch.manual_seed(1337)


@dataclass
class GPT2Config:
    sequence_window_size: int = 1024
    # GPT-2 vocab_size of 50257; in training we often pad to a multiple of 64 for efficiency.
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True
    # Used by `from_pretrained(..., override_args={"dropout": ...})`
    dropout: float = 0.0


# gpt2 model


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch LayerNorm can't do bias=False directly)."""

    def __init__(self, ndim: int, *, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )  # [Q|K|V] fused for speed
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        # Flash attention is available in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            # Causal mask buffer (matches HF naming: `attn.bias`), not a Parameter.
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.sequence_window_size, config.sequence_window_size)
                ).view(1, 1, config.sequence_window_size, config.sequence_window_size),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q, K, V = self.c_attn(x).split(
            self.n_embd, dim=2
        )  # (B, T, 3 * D) -> 3x (B, T, D)
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        if self.flash:
            # (B, H, T, D) x (B, H, T, D) -> (B, H, T, D)
            Y = torch.nn.functional.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (Q @ K.transpose(-2, -1)) * (
                self.head_dim**-0.5
            )  # (B, H, T, T) -- attention scores
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = att.softmax(dim=-1)
            att = self.attn_dropout(att)
            Y = att @ V  # (B, H, T, D) -- weighted sum of values
        Y = Y.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D) -- merge heads
        return self.resid_dropout(self.c_proj(Y))


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = GPT2Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.sequence_window_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: use token embedding weights for lm_head.
        self.transformer["wte"].weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        # Special scaled init to residual projections, per GPT-2 paper.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer["wpe"].weight.numel()
        return n_params

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        if T > self.config.sequence_window_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}, sequence_window_size is only {self.config.sequence_window_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        tok_emb = self.transformer["wte"](x)  # (B, T, D)
        pos_emb = self.transformer["wpe"](pos)  # (T, D)
        h = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            h = block(h)
        h = self.transformer["ln_f"](h)
        logits = self.lm_head(h)  # (B, T, V)

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        *,
        cfg: SamplingConfig | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper around the standalone `generate` function."""
        return generate(self, x, max_new_tokens=max_new_tokens, cfg=cfg)

    def crop_sequence_window_size(self, sequence_window_size: int) -> None:
        """Model surgery: shrink the context window (pos-emb + causal mask)."""
        if sequence_window_size > self.config.sequence_window_size:
            raise ValueError(
                f"sequence_window_size must be <= {self.config.sequence_window_size}, got {sequence_window_size}"
            )
        self.config.sequence_window_size = int(sequence_window_size)
        self.transformer["wpe"].weight = nn.Parameter(
            self.transformer["wpe"].weight[:sequence_window_size]
        )
        for block in self.transformer["h"]:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :sequence_window_size, :sequence_window_size]

    @classmethod
    def from_pretrained(
        cls, model_type: str, override_args: dict | None = None
    ) -> "GPT2Model":
        """Load GPT-2 weights from HuggingFace into this minimal implementation."""
        if override_args is None:
            override_args = {}
        if any(k != "dropout" for k in override_args):
            raise ValueError(
                f"Only 'dropout' can be overridden, got: {list(override_args.keys())}"
            )

        sizes = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        if model_type not in sizes:
            raise ValueError(
                f"Unknown model_type={model_type!r}, expected one of: {sorted(sizes)}"
            )

        # HF/OpenAI checkpoint invariants for GPT-2
        config = GPT2Config(
            **sizes[model_type],
            vocab_size=50257,
            sequence_window_size=1024,
            bias=True,
            dropout=float(override_args.get("dropout", 0.0)),
        )

        from transformers import GPT2LMHeadModel

        model = cls(config)
        sd = model.state_dict()
        sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()

        # HF GPT-2 uses a Conv1D module for some projections; our Linear needs transposed weights.
        needs_T = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )
        skip_suffixes = (".attn.bias", ".attn.masked_bias")  # HF buffers only

        with torch.no_grad():
            for k, v in sd_hf.items():
                if k.endswith(skip_suffixes):
                    continue
                if k not in sd:
                    raise KeyError(f"Unexpected HF key not present in our model: {k}")
                if k.endswith(needs_T):
                    if v.t().shape != sd[k].shape:
                        raise ValueError(
                            f"shape mismatch for {k}: hf{tuple(v.shape)} -> {tuple(v.t().shape)} vs ours{tuple(sd[k].shape)}"
                        )
                    sd[k].copy_(v.t())
                else:
                    if v.shape != sd[k].shape:
                        raise ValueError(
                            f"shape mismatch for {k}: hf{tuple(v.shape)} vs ours{tuple(sd[k].shape)}"
                        )
                    sd[k].copy_(v)

        return model

    def configure_optimizers(
        self,
        *,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay only on 2D+ params (matmuls + embeddings)."""
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = bool(fused_available and device_type == "cuda")
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

    def estimate_mfu(self, *, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate MFU in units of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L = cfg.n_layer
        H = cfg.n_head
        Q = cfg.n_embd // cfg.n_head
        T = cfg.sequence_window_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bf16 peak FLOPS (312 TFLOPS)
        return flops_achieved / flops_promised


if __name__ == "__main__":
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, how are you?"
    tokens = tokenizer.encode(text)
    x = torch.tensor(tokens).unsqueeze(0)  # (T,) -> (1, T)
    print(model.generate(x, 10, cfg=SamplingConfig(strategy="top_k")))
    print(tokenizer.decode(model.generate(x, 10, cfg=SamplingConfig(strategy="top_k"))[0].tolist()))
