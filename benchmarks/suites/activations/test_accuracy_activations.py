"""
benchmarks/test_accuracy_activations.py — Accuracy Test for 8-bit Activations

Trains a simple CNN on MNIST to compare:
1. Standard AdamW (Baseline)
2. AdamW + SMO 8-bit Quantized Activations
"""

import argparse
import sys
from pathlib import Path
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._paths import add_project_root_to_path
add_project_root_to_path()

from smo.activations_8bit import wrap_model_activations


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def measure_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0  # On CPU we can't easily measure peak activation RAM


def run_experiment(use_smo_activations=False, epochs=1, device='cpu', seed=1234):
    set_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = SimpleCNN().to(device)
    if use_smo_activations:
        print("Applying SMO 8-bit Activation Wrapping...")
        model = wrap_model_activations(model, block_size=64)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"Training for {epochs} epoch(s)...")
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        print(f"Epoch {epoch}: Loss: {loss:.4f}, Test Acc: {accuracy:.2f}%")
    
    total_time = time.time() - start_time
    peak_mem = measure_memory_usage()
    
    return accuracy, total_time, peak_mem


def main():
    parser = argparse.ArgumentParser(description="Accuracy & memory test: AdamW vs SMO 8-bit activations on MNIST")
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    print("MNIST Accuracy & Memory Test: Baseline vs SMO 8-bit Activations")
    print("=" * 60)
    print(f"Epochs: {args.epochs}  Seed: {args.seed}")

    print("\n[1/2] Running Baseline (Standard AdamW)...")
    acc_base, time_base, mem_base = run_experiment(use_smo_activations=False, epochs=args.epochs, seed=args.seed)
    
    print("\n[2/2] Running SMO 8-bit Activations + AdamW...")
    acc_smo, time_smo, mem_smo = run_experiment(use_smo_activations=True, epochs=args.epochs, seed=args.seed)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Configuration':<25} | {'Accuracy':<10} | {'Memory (Peak)':<15} | {'Time'}")
    print("-" * 70)
    print(f"{'Standard AdamW':<25} | {acc_base:>9.2f}% | {mem_base:>10.2f} MB | {time_base:.2f}s")
    print(f"{'SMO 8-bit Activations':<25} | {acc_smo:>9.2f}% | {mem_smo:>10.2f} MB | {time_smo:.2f}s")
    
    mem_saved = mem_base - mem_smo
    mem_perc = (1 - mem_smo / mem_base) * 100 if mem_base > 0 else 0
    
    print("-" * 70)
    print(f"{'Difference':<25} | {acc_smo - acc_base:>9.2f}% | {mem_saved:>10.2f} MB ({mem_perc:.1f}%) | {time_smo - time_base:+.2f}s")
    print("\nNote: Memory metrics require CUDA to be accurate.")


if __name__ == "__main__":
    main()