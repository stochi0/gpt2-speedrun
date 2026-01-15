"""Training script for GPT-2 on FineWeb.

Examples:
  python3 main.py --device=auto --batch_size=8
  torchrun --standalone --nproc_per_node=4 main.py
"""

from __future__ import annotations

import sys

from config import TrainConfig, apply_cli_overrides
from trainer import Trainer


def main(argv: list[str]) -> None:
    Trainer(apply_cli_overrides(TrainConfig(), argv)).train()


if __name__ == "__main__":
    main(sys.argv)


