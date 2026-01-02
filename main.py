from dataclasses import dataclass
import torch
import torch.nn as nn

torch.manual_seed(1337)

# gpt2 config 

@dataclass
class GPT2Config:
    sequence_window_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# gpt2 model

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # [Q|K|V] -- fused QKV for speed up
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.bias = torch.tril(torch.ones(config.sequence_window_size, config.sequence_window_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, D = x.shape
            Q, K, V = self.c_attn(x).split(self.n_embd, dim=2)
            Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            attn_scores = Q @ K.transpose(-2, -1) * (self.head_dim ** -0.5)
            attn_scores = attn_scores.mask_fill(self.bias[:, :T, :T] == 0, float("-inf"))
            attn_scores = attn_scores.softmax(dim=-1)
            attn = attn_scores @ V
            attn = attn.transpose(1, 2).contiguous().view(B, T, D)
            return self.c_proj(attn)
   
class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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
        return x
