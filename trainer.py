"""Training utilities and trainer class."""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TrainConfig
from data import create_dataloaders
from model import GPT2Config, GPT2Model


@dataclass(frozen=True)
class DDPInfo:
    """Distributed training metadata."""

    ddp: bool
    rank: int
    local_rank: int
    world_size: int
    master_process: bool
    seed_offset: int


def resolve_device(device: str) -> str:
    """Resolve device string with fallbacks (auto -> cuda -> mps -> cpu)."""
    device = device.strip()

    def has_mps() -> bool:
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_built() and mps.is_available()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if has_mps():
            return "mps"
        return "cpu"

    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda:0"
        if has_mps():
            print("CUDA requested but not available; falling back to MPS.")
            return "mps"
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"

    if device.startswith("cuda:"):
        if torch.cuda.is_available():
            return device
        if has_mps():
            print(f"{device} requested but CUDA not available; falling back to MPS.")
            return "mps"
        print(f"{device} requested but CUDA not available; falling back to CPU.")
        return "cpu"

    if device == "mps":
        if has_mps():
            return "mps"
        print("MPS requested but not available; falling back to CPU.")
        return "cpu"

    if device == "cpu":
        return "cpu"

    return device


def setup_ddp(cfg: TrainConfig) -> tuple[TrainConfig, DDPInfo]:
    """Initialize DDP if running under torchrun, otherwise return single-process config."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if not ddp:
        return (
            cfg,
            DDPInfo(
                ddp=False,
                rank=0,
                local_rank=0,
                world_size=1,
                master_process=True,
                seed_offset=0,
            ),
        )

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA.")

    init_process_group(backend=cfg.backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    if cfg.gradient_accumulation_steps % world_size != 0:
        raise ValueError(
            f"gradient_accumulation_steps ({cfg.gradient_accumulation_steps}) "
            f"must be divisible by world_size ({world_size})."
        )

    new_cfg = TrainConfig(
        **{
            **{f.name: getattr(cfg, f.name) for f in fields(cfg)},
            "device": device,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps // world_size,
        }
    )

    return (
        new_cfg,
        DDPInfo(
            ddp=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            master_process=(rank == 0),
            seed_offset=rank,
        ),
    )


def setup_torch(cfg: TrainConfig, seed_offset: int):
    """Configure PyTorch settings and return autocast context + scaler."""
    torch.manual_seed(1337 + seed_offset)

    if "cuda" in cfg.device:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_type = "cuda"
    elif cfg.device == "mps":
        device_type = "mps"
    else:
        device_type = "cpu"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.dtype]

    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=("cuda" in cfg.device and cfg.dtype == "float16")
    )

    return device_type, ctx, scaler


def build_model(cfg: TrainConfig) -> tuple[GPT2Model, dict, int, float, dict | None]:
    """
    Build or load a GPT-2 model.
    
    Returns:
        (model, model_args, iter_num, best_val_loss, optimizer_state)
    """
    iter_num = 0
    best_val_loss = 1e9
    optimizer_state = None

    model_args = dict(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        sequence_window_size=cfg.sequence_window_size,
        bias=cfg.bias,
        dropout=cfg.dropout,
    )

    if cfg.init_from == "scratch":
        print("Initializing model from scratch")
        model = GPT2Model(
            GPT2Config(
                sequence_window_size=cfg.sequence_window_size,
                vocab_size=cfg.vocab_size,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_embd=cfg.n_embd,
                bias=cfg.bias,
                dropout=cfg.dropout,
            )
        )
    elif cfg.init_from == "resume":
        print(f"Resuming training from {cfg.out_dir}")
        ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        checkpoint_model_args = checkpoint["model_args"]

        for k in ["n_layer", "n_head", "n_embd", "sequence_window_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]

        model = GPT2Model(
            GPT2Config(
                sequence_window_size=int(model_args["sequence_window_size"]),
                vocab_size=int(model_args["vocab_size"]),
                n_layer=int(model_args["n_layer"]),
                n_head=int(model_args["n_head"]),
                n_embd=int(model_args["n_embd"]),
                bias=bool(model_args["bias"]),
                dropout=float(model_args["dropout"]),
            )
        )

        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        iter_num = int(checkpoint["iter_num"])
        best_val_loss = float(checkpoint["best_val_loss"])
        optimizer_state = checkpoint.get("optimizer")
    elif cfg.init_from.startswith("gpt2"):
        print(f"Initializing from HuggingFace GPT-2: {cfg.init_from}")
        model = GPT2Model.from_pretrained(
            cfg.init_from, override_args={"dropout": cfg.dropout}
        )
        model_args["n_layer"] = model.config.n_layer
        model_args["n_head"] = model.config.n_head
        model_args["n_embd"] = model.config.n_embd
        model_args["sequence_window_size"] = model.config.sequence_window_size
        model_args["bias"] = model.config.bias
        model_args["vocab_size"] = model.config.vocab_size
    else:
        raise ValueError(f"Unknown init_from: {cfg.init_from!r}")

    model.to(cfg.device)
    return model, model_args, iter_num, best_val_loss, optimizer_state


def create_optimizer(cfg: TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create AdamW optimizer with weight decay only on 2D+ params."""
    decay_params = []
    nodecay_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay_params if p.dim() >= 2 else nodecay_params).append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        optim_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
    )


def get_lr(cfg: TrainConfig, it: int) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, data_iter, ctx, eval_iters: int) -> float:
    """Estimate loss over eval_iters batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            break
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(
    out_dir: str,
    raw_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: dict,
    iter_num: int,
    best_val_loss: float,
    config: dict,
):
    """Save training checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))


class Trainer:
    """Main training loop orchestrator."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = TrainConfig(
            **{
                **{f.name: getattr(cfg, f.name) for f in fields(cfg)},
                "device": resolve_device(cfg.device),
            }
        )
        self.cfg, self.ddp_info = setup_ddp(self.cfg)
        self.device_type, self.ctx, self.scaler = setup_torch(
            self.cfg, self.ddp_info.seed_offset
        )

        tokens_per_iter = (
            self.cfg.gradient_accumulation_steps
            * self.ddp_info.world_size
            * self.cfg.batch_size
            * self.cfg.sequence_window_size
        )
        print(f"tokens per iteration: {tokens_per_iter:,}")

        if self.ddp_info.master_process:
            os.makedirs(self.cfg.out_dir, exist_ok=True)

        # Data
        self.train_loader, self.val_loader = create_dataloaders(
            dataset_name=self.cfg.dataset,
            dataset_split=self.cfg.dataset_split,
            sequence_window_size=self.cfg.sequence_window_size,
            batch_size=self.cfg.batch_size,
            device=self.cfg.device,
            device_type=self.device_type,
        )
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

        # Model
        model, self.model_args, self.iter_num, self.best_val_loss, optimizer_state = (
            build_model(self.cfg)
        )

        self.optimizer = create_optimizer(self.cfg, model)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        if self.cfg.compile:
            try:
                print("compiling model... (takes ~minute)")
                model = torch.compile(model)
            except Exception as e:
                print(f"torch.compile failed: {e}")

        if self.ddp_info.ddp:
            model = DDP(model, device_ids=[self.ddp_info.local_rank])

        self.model = model
        self.raw_model = model.module if self.ddp_info.ddp else model

        # Logging
        if self.cfg.wandb_log and self.ddp_info.master_process:
            import wandb

            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config={f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)},
            )
            self.wandb = wandb
        else:
            self.wandb = None

    def train(self):
        """Run the training loop."""
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0

        while True:
            # Learning rate schedule
            lr = get_lr(self.cfg, self.iter_num) if self.cfg.decay_lr else self.cfg.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluation
            if self.iter_num % self.cfg.eval_interval == 0 and self.ddp_info.master_process:
                train_loss = estimate_loss(
                    self.model, self.train_iter, self.ctx, self.cfg.eval_iters
                )
                val_loss = estimate_loss(
                    self.model, self.val_iter, self.ctx, self.cfg.eval_iters
                )
                print(f"step {self.iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

                if self.wandb:
                    self.wandb.log(
                        {
                            "iter": self.iter_num,
                            "train/loss": train_loss,
                            "val/loss": val_loss,
                            "lr": lr,
                            "mfu": running_mfu * 100,
                        }
                    )

                if val_loss < self.best_val_loss or self.cfg.always_save_checkpoint:
                    self.best_val_loss = val_loss
                    if self.iter_num > 0:
                        save_checkpoint(
                            out_dir=self.cfg.out_dir,
                            raw_model=self.raw_model,
                            optimizer=self.optimizer,
                            model_args=self.model_args,
                            iter_num=self.iter_num,
                            best_val_loss=self.best_val_loss,
                            config={f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)},
                        )

            if self.iter_num == 0 and self.cfg.eval_only:
                break

            # Training step with gradient accumulation
            for micro_step in range(self.cfg.gradient_accumulation_steps):
                if self.ddp_info.ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.cfg.gradient_accumulation_steps - 1
                    )

                x, y = next(self.train_iter)
                with self.ctx:
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / self.cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

            if self.cfg.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if self.iter_num % self.cfg.log_interval == 0 and self.ddp_info.master_process:
                lossf = loss.item() * self.cfg.gradient_accumulation_steps
                print(
                    f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms"
                )

            self.iter_num += 1
            local_iter_num += 1

            if self.iter_num > self.cfg.max_iters:
                break

        if self.ddp_info.ddp:
            destroy_process_group()

