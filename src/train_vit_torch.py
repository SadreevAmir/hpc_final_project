#!/usr/bin/env python3
"""Idiomatic PyTorch training path for the MNIST pixel-token ViT."""

from __future__ import annotations

import argparse
import csv
import os
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, RandomSampler


@dataclass(frozen=True)
class Cfg:
    vocab: int = 256
    seq: int = 28 * 28
    layers: int = 2
    dim: int = 64
    heads: int = 4
    classes: int = 10


DEFAULT_PARAM_COUNT = 167_296


class MnistCsvDataset(Dataset):
    def __init__(self, path: Path, seq: int) -> None:
        self.pixels, self.labels = load_mnist_csv(path, seq)

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pixels[index], self.labels[index]


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.dim,
            num_heads=cfg.heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.dim, cfg.dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        with sdpa_kernel(SDPBackend.MATH):
            attn_out = self.attn(h, h, h, need_weights=False)[0]
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class PixelViT(nn.Module):
    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        if cfg.dim % cfg.heads:
            raise ValueError("dim must be divisible by heads")

        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab, cfg.dim)
        self.position_embedding = nn.Parameter(torch.empty(1, cfg.seq, cfg.dim))
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.head = nn.Linear(cfg.dim, cfg.classes, bias=False)
        self.apply(init_module)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.shape[1] != self.cfg.seq:
            raise ValueError(f"expected sequence length {self.cfg.seq}, got {pixels.shape[1]}")

        x = self.token_embedding(pixels) + self.position_embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def init_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)


def load_mnist_csv(path: Path, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    labels = array("q")
    pixels_raw = bytearray()
    with path.open(newline="") as fp:
        rows = csv.reader(fp)
        header = next(rows, None)
        if header is None or len(header) != seq + 1:
            raise ValueError(f"{path} must have label + {seq} pixel columns")

        for line_no, row in enumerate(rows, start=2):
            if len(row) != seq + 1:
                raise ValueError(f"{path}:{line_no} has {len(row)} columns")
            labels.append(int(row[0]))
            pixels_raw.extend(int(pixel) for pixel in row[1:])

    if not labels:
        raise ValueError(f"{path} contains no training samples")

    labels_tensor = torch.frombuffer(labels, dtype=torch.int64).clone()
    pixels_tensor = torch.frombuffer(pixels_raw, dtype=torch.uint8).clone().view(-1, seq)
    return pixels_tensor, labels_tensor


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def distributed_context(device_arg: str) -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda", local_rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    else:
        device = torch.device(device_arg)

    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def reduce_metrics(
    loss: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    world: int,
) -> tuple[float, float]:
    with torch.no_grad():
        count = torch.tensor(labels.numel(), device=labels.device, dtype=torch.float32)
        loss_sum = loss.detach() * count
        correct = (logits.argmax(dim=1) == labels).sum(dtype=torch.float32)
        if world > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return (loss_sum / count).item(), (correct / count).item()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch MNIST pixel-token ViT counterpart for train_vit.cu.",
    )
    parser.add_argument("csv_path", nargs="?", default="data/train.csv")
    parser.add_argument("steps", nargs="?", type=int, default=200)
    parser.add_argument("batch_size", nargs="?", type=int, default=8)
    parser.add_argument("lr", nargs="?", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-path", default="training_log.csv")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Cfg()
    rank, world, device = distributed_context(args.device)
    log_fp = None

    try:
        torch.manual_seed(args.seed)
        dataset = MnistCsvDataset(Path(args.csv_path), cfg.seq)
        sampler_rng = torch.Generator()
        sampler_rng.manual_seed(args.seed + 1000 * rank)
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=args.steps * args.batch_size,
            generator=sampler_rng,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

        model = PixelViT(cfg).to(device)
        param_count = count_parameters(model)
        if param_count != DEFAULT_PARAM_COUNT:
            raise AssertionError(f"default model parameter count drifted: {param_count}")
        train_model: nn.Module
        if world > 1:
            train_model = DDP(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
            )
        else:
            train_model = model
        optimizer = torch.optim.Adam(
            train_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if rank == 0:
            print(
                f"ranks={world} N={len(dataset)} B={args.batch_size} "
                f"T={cfg.seq} L={cfg.layers} D={cfg.dim} H={cfg.heads} "
                f"C={cfg.classes} params={param_count} "
                f"steps={args.steps} lr={args.lr:g} impl=torch device={device.type}",
                flush=True,
            )
            log_fp = Path(args.log_path).open("w", newline="")
            log_writer = csv.writer(log_fp)
            log_writer.writerow(("step", "elapsed_s", "loss", "accuracy"))
            log_fp.flush()
        else:
            log_writer = None

        sync_device(device)
        start = time.perf_counter()
        train_model.train()

        for step, (pixels, labels) in enumerate(loader, start=1):
            pixels = pixels.to(device=device, dtype=torch.long, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = train_model(pixels)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            should_log = step % args.log_every == 0 or step == args.steps
            if should_log:
                mean_loss, accuracy = reduce_metrics(loss, logits, labels, world)
                if rank == 0:
                    sync_device(device)
                    elapsed = time.perf_counter() - start
                    print(f"step {step:4d} | loss {mean_loss:.4f} | acc {accuracy:.3f}", flush=True)
                    if log_writer is not None and log_fp is not None:
                        log_writer.writerow(
                            (step, f"{elapsed:.2f}", f"{mean_loss:.4f}", f"{accuracy:.4f}")
                        )
                        log_fp.flush()

        sync_device(device)
        if world > 1:
            dist.barrier()
        elapsed = time.perf_counter() - start
        total_images = args.steps * args.batch_size * world
        if rank == 0:
            print(f"\nfinished: {args.steps} steps in {elapsed:.2f}s")
            print(
                f"throughput: {total_images / elapsed:.0f} img/s global  |  "
                f"{total_images / elapsed / world:.0f} img/s/GPU"
            )
    finally:
        if log_fp is not None:
            log_fp.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
