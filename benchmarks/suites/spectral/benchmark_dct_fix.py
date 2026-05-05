#!/usr/bin/env python3
"""
Targeted Benchmark: SMODCTPure (with smoothing fix) vs AdamW on CIFAR-10.
"""

import os
# Benchmark classification: family=end_to_end_training, category=diagnostic, status=canonical
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from benchmarks._paths import DATA_DIR, add_project_root_to_path

add_project_root_to_path()

from spectral.optim_dct_pure import SMODCTPure

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

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

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

def run_experiment(name, opt_fn, epochs=3):
    print(f"\nRunning: {name}")
    device = 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    
    model = CIFAR_CNN().to(device)
    optimizer = opt_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Time: {time.time()-start:.2f}s")
    return acc

if __name__ == '__main__':
    print("="*60)
    print("DCT Resurrected Benchmark (Gibbs Fix)")
    print("="*60)
    acc_adam = run_experiment("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3))
    acc_dct = run_experiment("SMODCTPure (Gibbs Fix)", lambda p: SMODCTPure(p, lr=1e-3, k_ratio=0.5))
    
    print("\n" + "="*60)
    print(f"FINAL ACCURACY COMPARISON:")
    print(f"AdamW: {acc_adam:.2f}%")
    print(f"DCT Pure: {acc_dct:.2f}%")
    print("="*60)
