# GPT-2 Speedrun

Clean, minimal GPT-2 implementation with training on FineWeb-Edu.

## Project Structure

```
gpt2-speedrun/
├── model.py       # GPT-2 architecture
├── sampling.py    # Text generation utilities
├── config.py      # Training configuration
├── data.py        # FineWeb dataset loader
├── trainer.py     # Training orchestration
└── main.py        # Main training entrypoint
```

## Quick Start

### Install Dependencies

```bash
uv sync
```

### Train on FineWeb-Edu

```bash
# Single GPU/MPS/CPU
python main.py --device=mps --batch_size=8 --compile=false

# Multi-GPU DDP (4 GPUs)
torchrun --standalone --nproc_per_node=4 main.py
```

### Configuration

All hyperparameters are in `config.py`. Override via CLI:

```bash
python main.py \
  --dataset=HuggingFaceFW/fineweb-edu \
  --dataset_split=sample-10BT \
  --batch_size=12 \
  --learning_rate=6e-4 \
  --max_iters=100000 \
  --device=cuda
```

### Device Support

- **CUDA**: Single or multi-GPU via DDP
- **MPS**: Apple Silicon (M1/M2/M3)
- **CPU**: Fallback for debugging

Use `--device=auto` to automatically select the best available device.

### Resume Training

```bash
python main.py --init_from=resume --out_dir=out
```

### Load Pretrained GPT-2

```bash
python main.py --init_from=gpt2  # or gpt2-medium, gpt2-large, gpt2-xl
```

## Architecture

- **model.py**: Minimal GPT-2 (multi-head attention, transformer blocks, LM head)
- **sampling.py**: Greedy, multinomial, top-k, top-p sampling strategies
- **data.py**: Streaming FineWeb dataset with GPT-2 tokenization
- **trainer.py**: Training loop, DDP setup, checkpointing, evaluation
- **config.py**: Single source of truth for all hyperparameters

## Notes

- FineWeb-Edu is streamed from HuggingFace, no local preprocessing needed
- Supports `torch.compile` for 2x+ speedup (PyTorch 2.0+)
- Automatic mixed precision (fp16/bf16) on CUDA
- Cosine LR schedule with linear warmup
