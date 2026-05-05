"""
benchmarks/debug_accuracy_drop.py — Debugging the Accuracy Loss

Tests components individually to find the culprit:
1. Baseline: AdamW
2. Squeezer Only: AdamW + SMO Activation Squeezer (8-bit Hooks)
3. Optimizer Only: SMO8bit Optimizer (k_ratio=0.25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smo import SMO8bit
from smo.activations_hooks import smo_squeezer

# Set device for DirectML
import torch_directml
device = torch_directml.device()

class SimpleCNN(nn.Module):
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

def train(model, loader, optimizer, squeezer):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with squeezer:
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

def run_test(name, use_smo_optimizer=False, use_squeezer=False):
    print(f"\n>> Running Test: {name}")
    torch.manual_seed(42)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    # Use a subset for speed
    train_set = torch.utils.data.Subset(train_set, range(10000))
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    
    model = SimpleCNN().to(device)
    
    if use_smo_optimizer:
        optimizer = SMO8bit(model.parameters(), lr=1e-3, k_ratio=0.25)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
    squeezer = smo_squeezer(enabled=use_squeezer)
    
    for epoch in range(1, 3):
        train(model, train_loader, optimizer, squeezer)
        acc = evaluate(model, test_loader)
        print(f"   Epoch {epoch}: Accuracy = {acc:.2f}%")
    
    return acc

def main():
    print(f"Running Debug Benchmark on {device}")
    
    res_base = run_test("Baseline (AdamW)", False, False)
    res_squeezer = run_test("Squeezer Only (Hooks)", False, True)
    res_smo = run_test("Optimizer Only (SMO8bit)", True, False)
    
    print("\n" + "="*40)
    print("DEBUG SUMMARY")
    print("="*40)
    print(f"Baseline:       {res_base:.2f}%")
    print(f"Squeezer Only:  {res_squeezer:.2f}%")
    print(f"Optimizer Only: {res_smo:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
