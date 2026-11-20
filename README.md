# GPT-2 Speedrun

A PyTorch GPT-2 implementation with Hugging Face weight loading, streaming
FineWeb-Edu data, checkpointing, and distributed data parallelism.

## Run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/stochi0/gpt2-speedrun.git
cd gpt2-speedrun
uv sync --locked
uv run python main.py --device=auto
```

Training streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`) and requires network
access. `auto` selects CUDA, then MPS, then CPU. CUDA enables mixed precision;
MPS and CPU run without autocast. Defaults and CLI overrides live in
[config.py](config.py); unknown configuration keys are rejected.

For four CUDA processes:

```sh
uv run torchrun --standalone --nproc_per_node=4 main.py --device=cuda
```

`gradient_accumulation_steps` must be divisible by the process count.

## Checkpoints and evaluation

Checkpoints contain model/optimizer state, model arguments, iteration, best loss,
and training configuration. The default destination is `out/ckpt.pt`.

```sh
uv run python main.py --init_from=resume --out_dir=out
uv run python main.py --eval_only=true --init_from=resume --out_dir=out
uv run python main.py --init_from=gpt2
```

Pretrained variants: `gpt2`, `gpt2-medium`, `gpt2-large`, and `gpt2-xl`.
Weights are downloaded when initializing from Hugging Face.

**Validation is not a held-out split:** training and validation datasets use the
same streaming construction. Interpret validation loss accordingly. This repository
does not report a measured training-speed result.

## Code

- [model.py](model.py): architecture and pretrained weight loading.
- [trainer.py](trainer.py): DDP, training, evaluation, and checkpointing.
- [data.py](data.py): streaming tokenization and batching.
- [sampling.py](sampling.py): generation strategies.

```sh
uv run ruff check .
```

## Reference

Run commands from the repository root after the [Setup](#run).

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

If you want to sample from a **trained checkpoint**, load the checkpoint and call `model.load_state_dict(...)` (see the [Checkpoints](#checkpoints-and-evaluation)).

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

## Citation

<!-- BEGIN generated citations -->

```bibtex
@software{bodadeprojectgpt2speedrun,
  author = {Bodade, Ayush},
  title = {{GPT-2 Speedrun}},
  url = {https://github.com/stochi0/gpt2-speedrun},
  note = {Software implementation; cite the original research separately. Specify the commit or version used.}
}
```

<!-- END generated citations -->
