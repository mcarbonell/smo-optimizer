"""
benchmarks/exhaustive_benchmark_local.py — Exhaustive Statistical Benchmark for SMO (Local Version)

Compares:
1. Baseline: Standard AdamW
2. Full SMO: SMO8bit Optimizer + SMO Activation Squeezer (8-bit)

Metrics:
- Accuracy (Mean + Std over 5 seeds)
- Peak RAM/VRAM RSS (MB)
- Training Time (s)

Datasets: MNIST, CIFAR-10
Uses DirectML for AMD GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import sys
import torch_directml
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smo import SMO8bit
from smo.activations_hooks import smo_squeezer

device = torch_directml.device()
process = psutil.Process(os.getpid())

def get_memory_mb():
    # RSS is a good proxy for integrated GPUs as they share system RAM
    return process.memory_info().rss / (1024 * 1024)

# --- Architectures ---

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- Training Logic ---

def train_one_epoch(model, loader, optimizer, squeezer_context):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with squeezer_context:
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

def run_benchmark():
    print(f"🚀 Running Exhaustive Benchmark on {device}")
    
    seeds = [42, 43, 44, 45, 46]
    datasets_to_test = ["MNIST", "CIFAR10"]
    configs = ["Baseline", "Full-SMO"]
    
    results = {d: {c: [] for c in configs} for d in datasets_to_test}
    
    for dataset_name in datasets_to_test:
        print(f"\n--- Testing Dataset: {dataset_name} ---")
        
        if dataset_name == "MNIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_set = datasets.MNIST('./data', train=False, transform=transform)
            model_cls = MNIST_CNN
            epochs = 2 
            batch_size = 128
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_set = datasets.CIFAR10('./data', train=False, transform=transform)
            model_cls = CIFAR10_CNN
            epochs = 3 
            batch_size = 128

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

        for config in configs:
            print(f"  Config: {config}")
            for seed in seeds:
                torch.manual_seed(seed)
                
                model = model_cls().to(device)
                
                if config == "Baseline":
                    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
                    squeezer = smo_squeezer(enabled=False)
                else:
                    optimizer = SMO8bit(model.parameters(), lr=1e-3, k_ratio=0.5)
                    squeezer = smo_squeezer(enabled=True)
                
                start_mem = get_memory_mb()
                peak_mem = start_mem
                
                start_time = time.time()
                for epoch in range(1, epochs + 1):
                    train_one_epoch(model, train_loader, optimizer, squeezer)
                    peak_mem = max(peak_mem, get_memory_mb())
                
                total_time = time.time() - start_time
                acc = evaluate(model, test_loader)
                
                results[dataset_name][config].append({
                    "acc": acc,
                    "mem": peak_mem,
                    "time": total_time
                })
                print(f"    Seed {seed}: Acc={acc:.2f}%, Mem={peak_mem:.1f}MB, Time={total_time:.1f}s")

    # --- Print Summary Table ---
    print("\n" + "="*90)
    print(f"{'Dataset':<10} | {'Config':<12} | {'Accuracy (Mean±Std)':<22} | {'RAM Peak':<12} | {'Time'}")
    print("-" * 90)
    
    for dname in datasets_to_test:
        for cname in configs:
            data = results[dname][cname]
            accs = [x['acc'] for x in data]
            mems = [x['mem'] for x in data]
            times = [x['time'] for x in data]
            
            acc_m, acc_s = np.mean(accs), np.std(accs)
            mem_avg = np.mean(mems)
            time_avg = np.mean(times)
            
            print(f"{dname:<10} | {cname:<12} | {acc_m:>7.2f}% ± {acc_s:>4.2f} | {mem_avg:>9.2f} MB | {time_avg:>6.1f}s")
    print("="*90)

if __name__ == "__main__":
    run_benchmark()
