"""Training configuration.

This repo keeps model-shape config names consistent with `model.py` (GPT2Config),
e.g. `sequence_window_size`, `vocab_size`, `n_layer`, `n_head`, `n_embd`, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters and system settings."""

    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    wandb_log: bool = False
    wandb_project: str = "gpt2-speedrun"
    wandb_run_name: str = "gpt2"

    # data
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_split: str = "sample-10BT"
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    # Model context length (matches GPT2Config.sequence_window_size)
    sequence_window_size: int = 1024

    # model
    # Keep names consistent with GPT2Config in `model.py`.
    vocab_size: int = 50304  # GPT-2 vocab rounded up for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # adamw optimizer
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # learning rate decay settings
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # DDP settings
    backend: str = "nccl"

    # system
    device: str = "cuda"
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16'
    compile: bool = True


def _coerce(value: str, *, like) -> object:
    """Coerce a string CLI value into the type of `like`."""
    if isinstance(like, bool):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid bool value: {value!r}")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, str):
        return value
    raise TypeError(f"Unsupported config type for overrides: {type(like)}")


def apply_cli_overrides(cfg: TrainConfig, argv: list[str]) -> TrainConfig:
    """
    Minimal CLI overrides: supports `--key=value` for keys in TrainConfig.
    Unknown keys raise to catch typos early.
    """
    cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    for arg in argv[1:]:
        if not arg.startswith("--") or "=" not in arg:
            continue
        key, value = arg[2:].split("=", 1)
        if key not in cfg_dict:
            raise KeyError(f"Unknown config key: {key!r}. Valid: {sorted(cfg_dict)}")
        cfg_dict[key] = _coerce(value, like=cfg_dict[key])
    return TrainConfig(**cfg_dict)

