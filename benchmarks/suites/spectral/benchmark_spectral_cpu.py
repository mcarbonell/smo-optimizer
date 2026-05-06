#!/usr/bin/env python3
"""
Benchmark: SMO (Spatial) vs SMOWalsh vs SMODCT on CIFAR-10 (CPU).
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from benchmarks._paths import DATA_DIR, RESULTS_DIR, add_project_root_to_path

add_project_root_to_path()

from smo.optim import SMO
from spectral.optim_walsh import SMOWalsh
from spectral.optim_dct import SMODCT
from spectral.optim_walsh_pure import SMOWalshPure
from spectral.optim_dct_pure import SMODCTPure


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_optimizer_memory(optimizer):
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total / (1024 ** 2)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

def run_experiment(optimizer_name, optimizer_fn, epochs=3, device='cpu', seed=1234):
    """Run full training with deterministic seeding."""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Running: {optimizer_name} (seed={seed})")
    print(f"{'='*60}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    model = CIFAR_CNN().to(device)
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    results = {'optimizer': optimizer_name, 'epochs': []}
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        
        results['epochs'].append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2),
            'test_acc': round(test_acc, 2),
            'time': round(epoch_time, 2)
        })
        
        print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    opt_mem = get_optimizer_memory(optimizer)
    
    results['total_time'] = round(total_time, 2)
    results['optimizer_memory_mb'] = round(opt_mem, 2)
    results['final_test_acc'] = results['epochs'][-1]['test_acc']
    results['seed'] = seed  # deterministic seeding
    
    print(f"\nFinal Results:")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Final test accuracy: {results['final_test_acc']:.2f}%")
    print(f"  Optimizer state memory: {opt_mem:.2f} MB")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Spectral optimizers benchmark on CIFAR-10 (CPU)")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    args = parser.parse_args()

    device = 'cpu'
    epochs = args.epochs
    seed = args.seed
    
    print("="*60)
    print(f"Spectral Optimizers CPU Benchmark ({epochs} Epochs, seed={seed})")
    print("="*60)
    
    res_adam = run_experiment("Standard AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3), epochs, device, seed)
    res_walsh_pure = run_experiment("SMOWalsh (Pure FWHT) k=0.5", lambda p: SMOWalshPure(p, lr=1e-3, k_ratio=0.5), epochs, device, seed)
    res_dct_pure = run_experiment("SMODCT (Pure) k=0.5", lambda p: SMODCTPure(p, lr=1e-3, k_ratio=0.5), epochs, device, seed)
    res_walsh = run_experiment("SMOWalsh (Hybrid) k=0.5", lambda p: SMOWalsh(p, lr=1e-3, k_ratio=0.5), epochs, device, seed)
    res_dct = run_experiment("SMODCT (Hybrid) k=0.5", lambda p: SMODCT(p, lr=1e-3, k_ratio=0.5), epochs, device, seed)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY (3 EPOCHS)")
    print("="*60)
    
    for r in [res_adam, res_walsh_pure, res_dct_pure, res_walsh, res_dct]:
        print(f"\n{r['optimizer']}:")
        print(f"  Accuracy: {r['final_test_acc']:.2f}%")
        print(f"  Memory:   {r['optimizer_memory_mb']:.2f} MB")
        print(f"  Time:     {r['total_time']:.2f} s")

if __name__ == '__main__':
    main()
