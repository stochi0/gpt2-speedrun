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

## Code and reference

- [model.py](model.py): architecture and pretrained weight loading.
- [trainer.py](trainer.py): DDP, training, evaluation, and checkpointing.
- [data.py](data.py): streaming tokenization and batching.
- [sampling.py](sampling.py): generation strategies.
- [Training reference](docs/reference.md): sampling, configuration, seeds, and troubleshooting.

```sh
uv run ruff check .
```

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
