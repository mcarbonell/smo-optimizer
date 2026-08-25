#!/usr/bin/env python3
"""
Wide-CNN probe: does the conv-pooling negative replicate beyond the 620k toy CNN?

CIFAR_CNN recipe (batch 128, lr 1e-3, Dropout 0.3, no scheduler) with channels
x2 (~2.5M params, conv-dominated). One config per invocation; records land in
benchmarks/results/wide_cnn_probe/.

Measured (10 epochs, seed 1234, CPU):
    adam          74.61
    smo_dense     76.11   <- SMO-k0.5 BEATS Adam once the CNN is bigger
    smo_conv      69.79   <- conv-pooling negative replicates (-6.3 vs dense)

Usage:
    python -m benchmarks.suites.training.benchmark_cifar10_wide --config smo_conv --epochs 10
"""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from benchmarks._paths import DATA_DIR, RESULTS_DIR
from benchmarks.suites.comparison.t4_memory_benchmark import persistent_state_mb
from smo import SMO


class WideCNN(nn.Module):
    """CIFAR_CNN with channels x2: conv-dominated (~2.5M params)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Wide-CNN probe: SMO conv-pooling replication test")
    parser.add_argument("--config", choices=["adam", "smo_dense", "smo_conv"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tfm_norm = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*tfm_norm),
    ])
    tfm_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*tfm_norm)])

    set_seed(args.seed)
    model = WideCNN()
    n_params = sum(p.numel() for p in model.parameters())

    if args.config == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        label = "adam"
        compress_conv = None
    elif args.config == "smo_dense":
        opt = SMO(model.parameters(), lr=1e-3, k_ratio=0.5)
        label = "smo_k0.5_dense"
        compress_conv = False
    else:
        opt = SMO(model.parameters(), lr=1e-3, k_ratio=0.5, compress_conv=True)
        label = "smo_k0.5_conv"
        compress_conv = True

    train_loader = DataLoader(
        datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=tfm_train),
        batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=tfm_test),
        batch_size=1000, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = tot_correct = seen = 0
        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.size(0)
            tot_correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(1) == y).sum().item()
        test_acc = 100 * correct / len(test_loader.dataset)
        history.append({"epoch": ep, "train_loss": round(tot_loss / seen, 4),
                        "train_acc": round(100 * tot_correct / seen, 2),
                        "test_acc": round(test_acc, 2)})
        print(f"[{label}] epoch {ep}/{args.epochs} loss {history[-1]['train_loss']:.4f} "
              f"train_acc {history[-1]['train_acc']:.2f}% test_acc {test_acc:.2f}%", flush=True)

    record = {
        "label": label,
        "benchmark_family": "end_to_end_training",
        "script_name": "benchmarks/suites/training/benchmark_cifar10_wide.py",
        "hardware": "CPU",
        "dataset": "CIFAR-10",
        "model": "WideCNN_x2",
        "params": n_params,
        "compress_conv": compress_conv,
        "seed": args.seed,
        "epochs": args.epochs,
        "precision": "fp32",
        "metrics": {
            "final_test_acc": history[-1]["test_acc"],
            "optimizer_state_mb": round(persistent_state_mb(opt), 2),
        },
        "history": history,
    }
    out_dir = RESULTS_DIR / "wide_cnn_probe"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"wide_probe_{label}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
