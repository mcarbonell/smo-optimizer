"""
benchmarks/exhaustive_benchmark.py — Exhaustive Statistical Benchmark for SMO

Compares:
1. Baseline: Standard AdamW
2. Full SMO: SMO8bit Optimizer + SMO Activation Squeezer (8-bit)

Metrics:
- Accuracy (Mean + Std over 5 seeds)
- Peak VRAM (MB)
- Training Time (s)

Datasets: MNIST, CIFAR-10
"""

# Benchmark classification: family=end_to_end_training, category=end_to_end, status=canonical
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

# Modal integration
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.5.0", "torchvision", "numpy")
    .add_local_dir("smo", remote_path="/root/smo")
)

app = modal.App("smo-exhaustive-benchmark")

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

def train_one_epoch(model, device, loader, optimizer, squeezer_context):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with squeezer_context:
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
        optimizer.step()

def evaluate(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

@app.function(image=image, gpu="A10G", timeout=3600)
def run_benchmark():
    sys.path.append("/root")
    from smo import SMO8bit
    from smo.activations_hooks import smo_squeezer
    
    device = "cuda"
    seeds = [42, 43, 44, 45, 46]
    datasets_to_test = ["MNIST", "CIFAR10"]
    configs = ["Baseline", "Full-SMO"]
    
    results = {d: {c: [] for c in configs} for d in datasets_to_test}
    
    for dataset_name in datasets_to_test:
        print(f"\n--- Testing Dataset: {dataset_name} ---")
        
        if dataset_name == "MNIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_set = datasets.MNIST('/tmp/data', train=True, download=True, transform=transform)
            test_set = datasets.MNIST('/tmp/data', train=False, transform=transform)
            model_cls = MNIST_CNN
            epochs = 3
            batch_size = 128
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_set = datasets.CIFAR10('/tmp/data', train=True, download=True, transform=transform)
            test_set = datasets.CIFAR10('/tmp/data', train=False, transform=transform)
            model_cls = CIFAR10_CNN
            epochs = 5
            batch_size = 128

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

        for config in configs:
            print(f"  Config: {config}")
            for seed in seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                model = model_cls().to(device)
                
                if config == "Baseline":
                    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
                    squeezer = smo_squeezer(enabled=False)
                else:
                    optimizer = SMO8bit(model.parameters(), lr=1e-3, k_ratio=0.25)
                    squeezer = smo_squeezer(enabled=True)
                
                start_time = time.time()
                for epoch in range(1, epochs + 1):
                    train_one_epoch(model, device, train_loader, optimizer, squeezer)
                
                total_time = time.time() - start_time
                acc = evaluate(model, device, test_loader)
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                
                results[dataset_name][config].append({
                    "acc": acc,
                    "mem": peak_mem,
                    "time": total_time
                })
                print(f"    Seed {seed}: Acc={acc:.2f}%, Mem={peak_mem:.2f}MB, Time={total_time:.1f}s")

    # --- Print Summary Table ---
    print("\n" + "="*80)
    print(f"{'Dataset':<10} | {'Config':<12} | {'Accuracy (Mean±Std)':<20} | {'VRAM (Peak)':<12} | {'Time'}")
    print("-" * 80)
    
    for dname in datasets_to_test:
        for cname in configs:
            data = results[dname][cname]
            accs = [x['acc'] for x in data]
            mems = [x['mem'] for x in data]
            times = [x['time'] for x in data]
            
            acc_m, acc_s = np.mean(accs), np.std(accs)
            mem_avg = np.mean(mems)
            time_avg = np.mean(times)
            
            print(f"{dname:<10} | {cname:<12} | {acc_m:>6.2f}% ± {acc_s:>4.2f} | {mem_avg:>8.2f} MB | {time_avg:>6.1f}s")
    print("="*80)

if __name__ == "__main__":
    with app.run():
        run_benchmark.remote()
