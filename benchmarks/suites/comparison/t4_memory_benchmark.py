#!/usr/bin/env python3
"""
T4/GPU memory benchmark: SMO vs AdamW vs bitsandbytes 8-bit AdamW.

Designed for a single 16 GB GPU (Colab/Kaggle T4) but works on any device.
Two suites:

  gpt  — char-level transformer on tiny-shakespeare (linear-heavy, auto-download)
  vit  — small ViT on CIFAR-10 (conv patch-embed + transformer blocks)

Per optimizer it reports:
  - peak CUDA memory (allocated/reserved) during training
  - persistent optimizer-state bytes (what a checkpoint stores)
  - compressed-parameter coverage (% of params SMO actually compresses)
  - task metric (val loss for gpt, test accuracy for vit)
  - throughput and wall time
  - status (ok / oom) — OOM is recorded per optimizer, never fatal

Fairness notes:
  - weight_decay=0 for every optimizer (SMO implements L2-in-gradient,
    not decoupled weight decay, so decoupled variants are not comparable)
  - identical LR schedule (cosine + warmup), grad clip, seed, batch order
  - --amp uses fp16 autocast for forward/backward only; optimizer states
    stay fp32 for ALL optimizers (safe for SMO moment math on T4/Turing)

Usage (T4 quality run):
  python -m benchmarks.suites.comparison.t4_memory_benchmark \
      --suite gpt --steps 1000 --amp --seed 1234
  python -m benchmarks.suites.comparison.t4_memory_benchmark \
      --suite vit --epochs 3 --amp --seed 1234

Killer-demo probe (model sized so fp32 AdamW may OOM on 16 GB):
  python -m benchmarks.suites.comparison.t4_memory_benchmark \
      --suite gpt --d_model 1024 --layers 24 --block_size 512 --batch 16 --amp

CPU smoke (validates plumbing anywhere):
  python -m benchmarks.suites.comparison.t4_memory_benchmark \
      --suite gpt --device cpu --steps 10 --d_model 64 --layers 2 \
      --heads 4 --block_size 64 --batch 8 --eval_interval 5
"""

import argparse
import gc
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks._paths import DATA_DIR
from benchmarks.results_utils import make_run_record, write_benchmark_bundle
from smo import SMO, SMO8bit

try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
OPTIMIZER_NAMES = ["adamw", "bnb8bit", "sgdm", "smo", "smo8bit"]


def set_seed(seed: int):
    torch.manual_seed(seed)
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(name: str, model: nn.Module, lr: float, k_ratio: float, protect_output: bool = False, low_peak: bool = False, permute_basis: bool = False):
    """Build an optimizer; for SMO variants optionally exclude embedding/head params."""
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if name == "bnb8bit":
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes not installed: pip install bitsandbytes")
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.999))
    if name == "sgdm":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name in ("smo", "smo8bit"):
        cls = SMO if name == "smo" else SMO8bit
        kwargs = {}
        if name == "smo8bit" and low_peak:
            kwargs["low_peak"] = True
        if name == "smo8bit" and permute_basis:
            kwargs["permute_basis"] = True
        if not protect_output:
            return cls(model.parameters(), lr=lr, k_ratio=k_ratio, **kwargs)
        protected, regular = [], []
        for pname, p in model.named_parameters():
            (protected if ("emb" in pname or "head" in pname) else regular).append(p)
        groups = [
            *([{"params": regular, "compress": True}] if regular else []),
            *([{"params": protected, "compress": False}] if protected else []),
        ]
        print(f"  protect_output: {sum(p.numel() for p in protected):,} protected / "
              f"{sum(p.numel() for p in regular):,} compressed params")
        return cls(groups, lr=lr, k_ratio=k_ratio, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def _iter_tensors_unique(obj, seen=None):
    """Yield every distinct tensor reachable through dicts/lists/tuples.

    bitsandbytes >= 0.4x nests quantized state inside
    __bnb_optimizer_quant_state__ dicts, so a shallow scan undercounts.
    """
    if seen is None:
        seen = set()
    if isinstance(obj, torch.Tensor):
        if id(obj) not in seen:
            seen.add(id(obj))
            yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors_unique(value, seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors_unique(value, seen)


def persistent_state_mb(optimizer) -> float:
    """Bytes reachable through state_dict() = what a checkpoint stores."""
    total = sum(t.numel() * t.element_size() for t in _iter_tensors_unique(optimizer.state_dict().get("state", {})))
    return total / (1024**2)


def compressed_coverage(model: nn.Module) -> float:
    """Fraction of parameters living in tensors SMO compresses (2D, both dims >= 32)."""
    covered = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.dim() == 2 and p.shape[0] >= 32 and p.shape[1] >= 32:
            covered += p.numel()
    return 100.0 * covered / max(1, total)


def build_schedule(optimizer, warmup: int, total_steps: int):
    warmup = max(1, min(warmup, total_steps // 2))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Suite: char-level GPT on tiny-shakespeare
# ---------------------------------------------------------------------------


def load_shakespeare(data_dir: Path) -> torch.Tensor:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tiny_shakespeare.txt"
    if not path.exists():
        print(f"Downloading tiny-shakespeare to {path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Corpus: {len(text):,} chars, vocab={len(chars)}")
    return encoded


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(attn_out)
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, vocab: int, d_model: int, depth: int, heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, heads, dropout) for _ in range(depth))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), diagonal=1)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def run_gpt(args, opt_name: str, device: torch.device) -> dict:
    corpus = args._corpus
    vocab = int(corpus.max().item()) + 1
    model = CharGPT(vocab, args.d_model, args.layers, args.heads, args.block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    split = int(0.9 * len(corpus))
    train_data, val_data = corpus[:split], corpus[split:]

    def get_batch(data):
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch,))
        x = torch.stack([data[i : i + args.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + 1 + args.block_size] for i in ix]).to(device)
        return x, y

    optimizer = make_optimizer(opt_name, model, args.lr, args.k_ratio,
                               getattr(args, "protect_output", False),
                               getattr(args, "low_peak", False),
                               getattr(args, "permute_basis", False))
    scheduler = build_schedule(optimizer, warmup=100, total_steps=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    @torch.no_grad()
    def estimate(split_data, batches=20):
        model.eval()
        losses = []
        ctx = torch.autocast("cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda")
        with ctx:
            for _ in range(batches):
                x, y = get_batch(split_data)
                _, loss = model(x, y)
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    t_start = time.perf_counter()
    tokens_seen = 0
    final_val = float("nan")
    history = []
    for step in range(args.steps):
        x, y = get_batch(train_data)
        ctx = torch.autocast("cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda")
        with ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.amp and device.type == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tokens_seen += x.numel()
        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            final_val = estimate(val_data)
            history.append({"step": step + 1, "train_loss": round(loss.item(), 4), "val_loss": round(final_val, 4)})
            print(f"  [{opt_name}] step {step + 1}/{args.steps} train_loss {loss.item():.4f} val_loss {final_val:.4f}")
    wall = time.perf_counter() - t_start

    result = {
        "status": "ok",
        "params": n_params,
        "coverage_pct": round(compressed_coverage(model), 2),
        "persistent_state_mb": round(persistent_state_mb(optimizer), 2),
        "final_val_loss": round(final_val, 4),
        "tokens_per_s": round(tokens_seen / wall),
        "wall_s": round(wall, 1),
        "history": history,
    }
    del model, optimizer
    return result


# ---------------------------------------------------------------------------
# Suite: small ViT on CIFAR-10
# ---------------------------------------------------------------------------


def cifar_loaders(batch_size: int, limit_batches: int):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    if limit_batches > 0:
        from itertools import islice

        train_loader = list(islice(train_loader, limit_batches))
    return train_loader, test_loader, len(test_ds)


class VitBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.block = TransformerBlock(width, heads, dropout)

    def forward(self, x):
        return self.block(x, None)


class TinyViT(nn.Module):
    def __init__(self, img_size: int, patch: int, n_classes: int, width: int, depth: int, heads: int, dropout: float = 0.1):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches + 1, width))
        self.blocks = nn.ModuleList(VitBlock(width, heads, dropout) for _ in range(depth))
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_classes)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, targets=None):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls.expand(tokens.size(0), -1, -1), tokens), dim=1) + self.pos_emb
        for block in self.blocks:
            tokens = block(tokens)
        logits = self.head(self.ln_f(tokens[:, 0]))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


@torch.no_grad()
def evaluate_vit(model, test_loader, device, n_test, autocast_on: bool) -> float:
    model.eval()
    correct = 0
    ctx = torch.autocast("cuda", dtype=torch.float16, enabled=autocast_on)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with ctx:
            logits, _ = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    model.train()
    return 100.0 * correct / n_test


def run_vit(args, opt_name: str, device: torch.device) -> dict:
    train_loader, test_loader, n_test = args._loaders
    autocast_on = args.amp and device.type == "cuda"
    model = TinyViT(32, args.patch, 10, args.width, args.depth, args.heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = make_optimizer(opt_name, model, args.lr, args.k_ratio,
                               getattr(args, "protect_output", False),
                               getattr(args, "low_peak", False),
                               getattr(args, "permute_basis", False))
    steps_per_epoch = len(train_loader)
    scheduler = build_schedule(optimizer, warmup=max(1, steps_per_epoch // 5), total_steps=steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_on)

    model.train()
    t_start = time.perf_counter()
    images_seen = 0
    final_acc = 0.0
    history = []
    for epoch in range(args.epochs):
        running = 0.0
        for bi, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            ctx = torch.autocast("cuda", dtype=torch.float16, enabled=autocast_on)
            with ctx:
                _, loss = model(images, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if autocast_on:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            images_seen += images.size(0)
            running += loss.item()
        final_acc = evaluate_vit(model, test_loader, device, n_test, autocast_on)
        epoch_loss = running / max(1, bi + 1)
        history.append({"epoch": epoch + 1, "train_loss": round(epoch_loss, 4), "test_acc": round(final_acc, 2)})
        print(f"  [{opt_name}] epoch {epoch + 1}/{args.epochs} loss {epoch_loss:.4f} test_acc {final_acc:.2f}%")
    wall = time.perf_counter() - t_start

    result = {
        "status": "ok",
        "params": n_params,
        "coverage_pct": round(compressed_coverage(model), 2),
        "persistent_state_mb": round(persistent_state_mb(optimizer), 2),
        "final_test_acc": round(final_acc, 2),
        "images_per_s": round(images_seen / wall),
        "wall_s": round(wall, 1),
        "history": history,
    }
    del model, optimizer
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="T4 memory benchmark: SMO vs AdamW vs bnb-8bit")
    parser.add_argument("--suite", choices=["gpt", "vit"], required=True)
    parser.add_argument("--optimizers", type=str, default=",".join(OPTIMIZER_NAMES))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k_ratio", type=float, default=0.25)
    parser.add_argument("--protect_output", action="store_true",
                        help="SMO variants: keep embedding/head params on dense Adam moments")
    parser.add_argument("--low_peak", action="store_true",
                        help="SMO8bit: row-banded compress/update (no full-size temporaries)")
    parser.add_argument("--permute_basis", action="store_true",
                        help="SMO8bit: pool gradients in a random permuted basis (locality ablation)")
    parser.add_argument("--tag", type=str, default="", help="Suffix for result filenames (multi-seed / ablations)")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast for fwd/bwd (states stay fp32)")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--limit_batches", type=int, default=0, help="Cap train batches per epoch (smoke)")
    # gpt knobs
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=0, help="0 = suite default (gpt 32, vit 128)")
    parser.add_argument("--steps", type=int, default=1000)
    # vit knobs
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.batch == 0:
        args.batch = 32 if args.suite == "gpt" else 128
    set_seed(args.seed)

    opt_names = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    for name in list(opt_names):
        if name == "bnb8bit" and not HAS_BNB:
            print("WARNING: bitsandbytes unavailable, skipping bnb8bit (pip install bitsandbytes)")
            opt_names.remove(name)

    if args.suite == "gpt":
        corpus = load_shakespeare(DATA_DIR)
        args._corpus = corpus
        runner = run_gpt
        workload = {
            "d_model": args.d_model, "layers": args.layers, "heads": args.heads,
            "block_size": args.block_size, "batch": args.batch, "steps": args.steps,
        }
        metric_key = "final_val_loss"
    else:
        loaders = cifar_loaders(args.batch, args.limit_batches)
        args._loaders = loaders
        runner = run_vit
        workload = {"width": args.width, "depth": args.depth, "heads": args.heads,
                    "patch": args.patch, "epochs": args.epochs}
        metric_key = "final_test_acc"

    runs = []
    print(f"\n{'=' * 72}")
    print(f"T4 memory benchmark | suite={args.suite} device={device.type} amp={args.amp} seed={args.seed}")
    print(f"Optimizers: {opt_names}")
    print(f"{'=' * 72}")

    for name in opt_names:
        labels = {"adamw": "AdamW-fp32", "bnb8bit": "bnb-AdamW8bit", "sgdm": f"SGD-M lr={args.lr}",
                  "smo": f"SMO k={args.k_ratio}", "smo8bit": f"SMO-8bit k={args.k_ratio}"}
        label = labels[name]
        if name == "smo8bit" and args.low_peak:
            label += " lp"
        print(f"\n--- {label} ---")
        set_seed(args.seed)
        free_memory()
        try:
            result = runner(args, name, device)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  [{name}] OUT OF MEMORY — recorded")
            result = {"status": "oom"}
        if device.type == "cuda" and result.get("status") == "ok":
            result["_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
        free_memory()

        record = make_run_record(
            benchmark_family="gpu_memory_training",
            variant=label,
            script_name="benchmarks/suites/comparison/t4_memory_benchmark.py",
            hardware=device.type.upper(),
            backend=device.type,
            dataset={"gpt": "tiny-shakespeare", "vit": "CIFAR-10"}[args.suite],
            model={"gpt": "CharGPT", "vit": "TinyViT"}[args.suite],
            seed=args.seed if result.get("status") == "ok" else None,
            precision="fp16-autocast" if args.amp else "fp32",
            epochs=args.epochs if args.suite == "vit" else None,
            steps=args.steps if args.suite == "gpt" else None,
            metrics=result,
            extra={"workload": workload, "k_ratio": args.k_ratio, "metric_key": metric_key,
                   "low_peak": bool(args.low_peak), "protect_output": bool(args.protect_output),
                   "permute_basis": bool(args.permute_basis)},
        )
        runs.append(record)

    ok_runs = [r for r in runs if r["metrics"].get("status") == "ok"]
    baseline = next((r for r in ok_runs if r["variant"] == "AdamW-fp32"), None)

    print(f"\n{'=' * 72}")
    print(f"{'SUMMARY':^72}")
    print(f"{'=' * 72}")
    header = f"{'optimizer':<20} {'peak_alloc':>11} {'state_MB':>9} {'metric':>10} {'throughput':>12} {'status':>7}"
    print(header)
    for r in runs:
        m = r["metrics"]
        if m.get("status") != "ok":
            print(f"{r['variant']:<20} {'—':>11} {'—':>9} {'—':>10} {'—':>12} {'OOM':>7}")
            continue
        metric = m.get(metric_key, float("nan"))
        thr = m.get("tokens_per_s") or m.get("images_per_s")
        thr_s = f"{thr:,}/s" if thr else "—"
        peak = m.get("_peak_alloc_mb")
        print(f"{r['variant']:<20} {str(peak or '—'):>11} {m['persistent_state_mb']:>9} {metric:>10} {thr_s:>12} {'ok':>7}")

    if baseline is not None and len(ok_runs) > 1:
        base_state = baseline["metrics"]["persistent_state_mb"]
        print(f"\nBaseline AdamW-fp32 persistent state: {base_state} MB")
        for r in ok_runs:
            if r is baseline:
                continue
            s = r["metrics"]["persistent_state_mb"]
            savings = (1 - s / base_state) * 100 if base_state > 0 else 0.0
            print(f"  {r['variant']}: {s} MB ({savings:.1f}% reduction)")

    suffix = f"_{args.tag}" if args.tag else ""
    aggregate, paths = write_benchmark_bundle(
        aggregate_filename=f"t4_{args.suite}_memory_results{suffix}.json",
        suite_name=f"t4_gpu_memory_{args.suite}{suffix}",
        benchmark_family="gpu_memory_training",
        summary={"suite": args.suite, "workload": workload, "amp": args.amp, "seed": args.seed},
        runs=runs,
    )
    print(f"\nResults saved to {aggregate}")
    print(f"Per-run files: {[p.name for p in paths]}")


if __name__ == "__main__":
    main()
