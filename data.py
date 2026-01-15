"""Dataset utilities for training."""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
import tiktoken


class FineWebDataset(IterableDataset):
    """
    Streaming FineWeb dataset with GPT-2 tokenization.
    
    Yields sequences of `sequence_window_size + 1` tokens for autoregressive training.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        sequence_window_size: int,
        *,
        streaming: bool = True,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.sequence_window_size = sequence_window_size
        self.streaming = streaming
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Load the dataset in streaming mode
        self.dataset = load_dataset(
            dataset_name,
            name=split,
            split="train",
            streaming=streaming,
        )

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            text = sample.get("text", "")
            if not text:
                continue
            
            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            buffer.extend(tokens)
            
            # Yield complete blocks
            while len(buffer) >= self.sequence_window_size + 1:
                chunk = buffer[: self.sequence_window_size + 1]
                buffer = buffer[self.sequence_window_size + 1 :]
                yield torch.tensor(chunk, dtype=torch.long)


def create_dataloaders(
    dataset_name: str,
    dataset_split: str,
    sequence_window_size: int,
    batch_size: int,
    device: str,
    device_type: str,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for FineWeb.
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = FineWebDataset(
        dataset_name=dataset_name,
        split=dataset_split,
        sequence_window_size=sequence_window_size,
    )
    
    # For validation, we use a small held-out portion (simulate with same data for now)
    val_dataset = FineWebDataset(
        dataset_name=dataset_name,
        split=dataset_split,
        sequence_window_size=sequence_window_size,
    )
    
    def collate_fn(batch):
        """Move batch to device and split into x, y."""
        batch = torch.stack(batch)  # (B, T+1)
        x = batch[:, :-1]  # (B, T)
        y = batch[:, 1:]   # (B, T)
        
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Keep 0 for streaming datasets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    return train_loader, val_loader

