from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# gpt2 config 

@dataclass
class GPT2Config:
    sequence_window_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True
    # Used by `from_pretrained(..., override_args={"dropout": ...})`
    dropout: float = 0.0


# gpt2 model

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # [Q|K|V] fused for speed
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # Causal mask buffer (matches HF naming: `attn.bias`), not a Parameter.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.sequence_window_size, config.sequence_window_size)).view(
                1, 1, config.sequence_window_size, config.sequence_window_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q, K, V = self.c_attn(x).split(self.n_embd, dim=2)  # (B, T, 3 * D) -> 3x (B, T, D)
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        att = (Q @ K.transpose(-2, -1)) * (self.head_dim**-0.5)  # (B, H, T, T) -- attention scores
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)

        Y = att @ V  # (B, H, T, D) -- weighted sum of values
        Y = Y.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D) -- merge heads
        return self.c_proj(Y)
   
class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))

class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
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
                "h": nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape 
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        tok_emb = self.transformer["wte"](x)
        pos_emb = self.transformer["wpe"](pos)
        x = tok_emb + pos_emb
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: dict | None = None) -> "GPT2Model":
        """Load GPT-2 weights from HuggingFace into this minimal implementation."""
        if override_args is None:
            override_args = {}
        if any(k != "dropout" for k in override_args):
            raise ValueError(f"Only 'dropout' can be overridden, got: {list(override_args.keys())}")

        sizes = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        if model_type not in sizes:
            raise ValueError(f"Unknown model_type={model_type!r}, expected one of: {sorted(sizes)}")

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
                        raise ValueError(f"shape mismatch for {k}: hf{tuple(v.shape)} -> {tuple(v.t().shape)} vs ours{tuple(sd[k].shape)}")
                    sd[k].copy_(v.t())
                else:
                    if v.shape != sd[k].shape:
                        raise ValueError(f"shape mismatch for {k}: hf{tuple(v.shape)} vs ours{tuple(sd[k].shape)}")
                    sd[k].copy_(v)

        return model

# sampling methods

@dataclass
class SamplingConfig:
    strategy: Literal["greedy", "multinomial", "top_k", "top_p"] = "multinomial"
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

# top-k sampling
def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    v, _ = torch.topk(logits, k, dim=-1)
    cutoff = v[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))

# top-p sampling    
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

if __name__ == "__main__":
    model = GPT2Model.from_pretrained("gpt2")
    print(model)
    print(model.generate(torch.tensor([[1, 2, 3]]), 10))