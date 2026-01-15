## GPT-2 Speedrun

 GPT-2 training with:
- a **from-scratch GPT-2** implementation that can **load HuggingFace GPT-2 weights**
- a **single-file trainer** with checkpointing + cosine LR schedule
- a **streaming FineWeb-Edu** dataloader using GPT-2 (tiktoken) tokenization

This repo is intentionally small and “research README”-style: the README is meant to be a spec for what the code actually does.

### Project layout

```
gpt2-speedrun/
├── main.py         # Entry point: build config from CLI and call Trainer.train()
├── config.py       # TrainConfig dataclass + strict --key=value overrides
├── trainer.py      # DDP setup, device/dtype setup, training loop, eval, checkpoints
├── data.py         # Streaming FineWeb-Edu IterableDataset + DataLoader collate
├── model.py        # GPT-2 architecture (+ HF weight import)
├── sampling.py     # Greedy/multinomial/top-k/top-p sampling utilities
├── pyproject.toml  # Dependencies (uv-managed)
├── uv.lock         # Locked dependency set
└── out/ckpt.pt      # (created at runtime) checkpoint output directory
```

### Installation (uv)

Requirements:
- **Python**: >= 3.11 (see `pyproject.toml`)
- **Package manager**: `uv`

Install:

```bash
uv sync
```

### Running training

The training entrypoint is `main.py`. Configuration is defined in `config.py` and overridden via `--key=value` CLI flags (unknown keys raise an error to catch typos).

Single process (CUDA / MPS / CPU):

```bash
python main.py --device=auto
```

Common overrides:

```bash
python main.py \
  --device=cuda \
  --batch_size=12 \
  --gradient_accumulation_steps=40 \
  --sequence_window_size=1024 \
  --learning_rate=6e-4 \
  --max_iters=600000 \
  --compile=true
```

Multi-GPU DDP (CUDA only):

```bash
torchrun --standalone --nproc_per_node=4 main.py --device=cuda
```

Notes:
- DDP is enabled automatically when `torchrun` sets `RANK/LOCAL_RANK/WORLD_SIZE`.
- DDP **requires CUDA**; `trainer.py` will raise if CUDA is unavailable.
- `gradient_accumulation_steps` must be divisible by `world_size` (enforced in `trainer.py`).

### Checkpointing and resuming

Checkpoints are saved to `--out_dir` (default: `out`) as:
- `out/ckpt.pt`

To resume:

```bash
python main.py --init_from=resume --out_dir=out
```

Checkpoint contents (`trainer.py:save_checkpoint`):
- `model`: raw model `state_dict()`
- `optimizer`: optimizer `state_dict()`
- `model_args`: shape/arch args needed to rebuild the model
- `iter_num`, `best_val_loss`
- `config`: the full training config dict at save time

### Initializing from HuggingFace GPT-2 weights

You can start from OpenAI/HF GPT-2 weights via `--init_from=gpt2*`:

```bash
python main.py --init_from=gpt2
```

Supported values (see `model.py`):
- `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`

Implementation detail: HuggingFace GPT-2 uses a `Conv1D` module for some projections; `model.py:GPT2Model.from_pretrained` transposes those weights into `nn.Linear` format.

### Evaluation mode

`--eval_only=true` runs the eval block at step 0 and exits:

```bash
python main.py --eval_only=true --init_from=resume --out_dir=out
```

### Sampling / generation

Generation utilities live in `sampling.py` and are also exposed via `GPT2Model.generate(...)` in `model.py`.

Supported strategies (`SamplingConfig.strategy`):
- `greedy`
- `multinomial`
- `top_k`
- `top_p`

Minimal example (run as a script or in a REPL):

```python
import torch
import tiktoken

from model import GPT2Model
from sampling import SamplingConfig

tok = tiktoken.get_encoding("gpt2")
model = GPT2Model.from_pretrained("gpt2").eval()

prompt = "Hello, how are you?"
x = torch.tensor(tok.encode(prompt)).unsqueeze(0)
y = model.generate(x, max_new_tokens=64, cfg=SamplingConfig(strategy="top_p", top_p=0.9))
print(tok.decode(y[0].tolist()))
```

If you want to sample from a **trained checkpoint**, load the checkpoint and call `model.load_state_dict(...)` (see “Checkpoint format” above).

### Dataset: FineWeb-Edu streaming

`data.py` implements an `IterableDataset` that streams from HuggingFace Datasets:
- dataset: `HuggingFaceFW/fineweb-edu` (default)
- dataset config/name: `sample-10BT` (default; passed as `name=...`)
- split: always `"train"` (streaming)

Tokenization:
- Uses `tiktoken.get_encoding("gpt2")`
- Encodes with `allowed_special={"<|endoftext|>"}`

Batch construction:
- Dataset yields blocks of length `sequence_window_size + 1`.
- Collate stacks to `(B, T+1)` then splits:
  - inputs `x = batch[:, :-1]` (B, T)
  - targets `y = batch[:, 1:]` (B, T)

Important limitation (current behavior):
- Validation is currently **not a true held-out split**; `create_dataloaders` builds `val_dataset` the same way as `train_dataset` (see comment in `data.py`). Treat validation metrics accordingly.

### Model: GPT-2 (implementation details)

`model.py` contains a compact GPT-2 reimplementation:
- **Pre-norm** Transformer blocks: LN → attention/MLP → residual add
- **Causal self-attention**
  - Uses PyTorch **scaled dot-product attention** (`F.scaled_dot_product_attention`) when available (flash path)
  - Otherwise uses an explicit causal mask buffer (`tril(...)`)
- **MLP**: Linear → GELU → Linear
- **Weight tying**: token embedding `wte.weight` is shared with `lm_head.weight`
- **Initialization**
  - Linear/embedding weights `N(0, 0.02)`
  - Residual projection weights `c_proj.weight` get GPT-2 scaled init (`0.02 / sqrt(2 * n_layer)`)

### Training loop (what actually happens)

`trainer.py:Trainer` orchestrates:
- **Device resolution** (`resolve_device`)
  - `auto` → `cuda:0` if available → else `mps` if available → else `cpu`
  - `cuda` falls back to `mps`/`cpu` with a warning if CUDA is missing
- **DDP setup** (`setup_ddp`)
  - Initializes process group with `backend=cfg.backend` (default: `nccl`)
  - Sets device to `cuda:{LOCAL_RANK}`
  - Divides `gradient_accumulation_steps` by `world_size`
- **Precision**
  - Autocast is enabled **only on CUDA** (`torch.amp.autocast(device_type="cuda", dtype=...)`)
  - GradScaler is enabled **only for CUDA + float16**
  - On MPS/CPU, the code runs without autocast (nullcontext)
- **Loss**
  - Cross-entropy over flattened `(B*T, vocab)` logits and `(B*T,)` targets
- **Gradient accumulation**
  - Each micro-step loss is divided by `gradient_accumulation_steps`
  - In DDP, gradient sync is delayed until the last micro-step (`require_backward_grad_sync`)
- **Optimizer**
  - AdamW
  - Weight decay is applied only to parameters with `dim() >= 2` (matmul weights + embeddings)
- **LR schedule**
  - Linear warmup (`warmup_iters`)
  - Cosine decay to `min_lr` until `lr_decay_iters`
- **Checkpointing**
  - Runs eval every `eval_interval` on the master process
  - Saves if `val_loss` improves OR `always_save_checkpoint=true` (default true)

Throughput reporting:
- Trainer prints `tokens per iteration` as:
  \[
  \text{grad\_accum} \times \text{world\_size} \times \text{batch\_size} \times \text{sequence\_window\_size}
  \]

### Configuration reference

All keys live in `config.py:TrainConfig`. Override via `--key=value` (unknown keys error).

Selected fields:
- **I/O**: `out_dir`, `eval_interval`, `eval_iters`, `eval_only`, `always_save_checkpoint`, `init_from`
- **Data**: `dataset`, `dataset_split`, `batch_size`, `gradient_accumulation_steps`, `sequence_window_size`
- **Model**: `vocab_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`
- **Optim**: `learning_rate`, `weight_decay`, `beta1`, `beta2`, `grad_clip`
- **Schedule**: `decay_lr`, `warmup_iters`, `lr_decay_iters`, `min_lr`
- **System**: `device`, `dtype`, `compile`
- **W&B**: `wandb_log`, `wandb_project`, `wandb_run_name`

### Reproducibility notes

- Training seed is set in `trainer.py` as `torch.manual_seed(1337 + seed_offset)`, where `seed_offset = rank` under DDP.
- `model.py` also sets `torch.manual_seed(1337)` at import time (model init reproducibility).

### Troubleshooting

- **DDP error about accumulation steps**: ensure `gradient_accumulation_steps % world_size == 0`.
- **No internet / dataset download failures**: streaming FineWeb requires HuggingFace dataset access at runtime.
- **MPS device**: use `--device=mps` (or `--device=auto`); autocast is not used on MPS in the current code.
- **`torch.compile` issues**: set `--compile=false` if compile fails on your environment.

### Development

- Lint (ruff):

```bash
uv run ruff check .
```
