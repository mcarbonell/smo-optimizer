"""
benchmarks/test_accuracy_delta.py — Accuracy Test for Delta-Encoded Activations

Trains a simple CNN on MNIST to compare:
1. Standard AdamW (Baseline)
2. AdamW + SMO 8-bit Activations
3. AdamW + SMO Delta-Encoded Activations (stored as float16)
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
from smo.activations_delta import wrap_model_delta


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
    return 0


def run_experiment(mode="baseline", epochs=1, device='cpu', seed=1234):
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
    
    if mode == "smo_8bit":
        print("Applying SMO 8-bit Activation Wrapping...")
        model = wrap_model_activations(model, block_size=64)
    elif mode == "smo_delta":
        print("Applying SMO Delta-Encoded Activation Wrapping...")
        model = wrap_model_delta(model)
    
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
    parser = argparse.ArgumentParser(description="Delta vs 8-bit activation accuracy test on MNIST")
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    print("MNIST Accuracy & Memory Test: Baseline vs SMO 8-bit vs SMO Delta")
    print("=" * 60)
    print(f"Epochs: {args.epochs}  Seed: {args.seed}")

    # 1. Baseline
    print("\n[1/3] Running Baseline (Standard AdamW)...")
    acc_base, time_base, mem_base = run_experiment(mode="baseline", epochs=args.epochs, seed=args.seed)
    
    # 2. SMO 8-bit Activations
    print("\n[2/3] Running SMO 8-bit Activations + AdamW...")
    acc_8bit, time_8bit, mem_8bit = run_experiment(mode="smo_8bit", epochs=args.epochs, seed=args.seed)
    
    # 3. SMO Delta
    print("\n[3/3] Running SMO Delta-Encoded Activations + AdamW...")
    acc_delta, time_delta, mem_delta = run_experiment(mode="smo_delta", epochs=args.epochs, seed=args.seed)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Configuration':<25} | {'Accuracy':<10} | {'Memory (Peak)':<15} | {'Time'}")
    print("-" * 70)
    print(f"{'Standard AdamW':<25} | {acc_base:>9.2f}% | {mem_base:>10.2f} MB | {time_base:.2f}s")
    print(f"{'SMO 8-bit Activations':<25} | {acc_8bit:>9.2f}% | {mem_8bit:>10.2f} MB | {time_8bit:.2f}s")
    print(f"{'SMO Delta Encoded':<25} | {acc_delta:>9.2f}% | {mem_delta:>10.2f} MB | {time_delta:.2f}s")
    
    print("\nNote: Memory metrics require CUDA to be accurate.")


if __name__ == "__main__":
    main()