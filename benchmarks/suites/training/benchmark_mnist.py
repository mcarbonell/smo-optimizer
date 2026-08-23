#!/usr/bin/env python3
"""
Benchmark: SMO (SWO) vs Standard Adam on MNIST.
Compares accuracy, loss, training time, and optimizer state memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import argparse
# Benchmark classification: family=end_to_end_training, category=end_to_end, status=canonical
from smo import SMO
from benchmarks._paths import DATA_DIR
from benchmarks.results_utils import make_run_record, write_benchmark_bundle


class SimpleCNN(nn.Module):
    """A small CNN suitable for MNIST."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_optimizer_memory(optimizer):
    """Calculate total optimizer state memory in MB."""
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
    
    for batch_idx, (data, target) in enumerate(loader):
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


def run_experiment(optimizer_name, optimizer_fn, epochs=5, device='cpu', seed=1234):
    """Run full training experiment with given optimizer factory."""
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Running: {optimizer_name} (seed={seed})")
    print(f"{'='*60}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    # Model
    model = SimpleCNN().to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Optimizer
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Memory before training
    opt_mem_before = get_optimizer_memory(optimizer)
    
    # Dummy forward to initialize states
    dummy = torch.zeros(1, 1, 28, 28).to(device)
    _ = model(dummy)
    
    # Training
    results = {
        'optimizer': optimizer_name,
        'parameters': param_count,
        'epochs': [],
        'train_time': 0,
        'seed': seed,  # deterministic seed
    }
    
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
    
    # Memory after training
    opt_mem_after = get_optimizer_memory(optimizer)
    
    results['total_time'] = round(total_time, 2)
    results['optimizer_memory_mb'] = round(opt_mem_after, 2)
    results['final_test_acc'] = results['epochs'][-1]['test_acc']
    
    print(f"\nFinal Results:")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Final test accuracy: {results['final_test_acc']:.2f}%")
    print(f"  Optimizer state memory: {opt_mem_after:.2f} MB")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MNIST end-to-end training: SMO vs Adam")
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs per variant')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', help='Torch device (cpu, cuda, ...)')
    args = parser.parse_args()

    device = args.device
    epochs = args.epochs
    seed = args.seed
    
    print("="*60)
    print("SWO Benchmark: SMO vs Standard Adam on MNIST")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {seed}")
    
    # Run Adam
    results_adam = run_experiment(
        "Standard Adam",
        lambda params: torch.optim.Adam(params, lr=1e-3),
        epochs=epochs,
        device=device,
        seed=seed
    )
    
    # Run SMO (k_ratio=0.25)
    results_swo = run_experiment(
        "SMO (k_ratio=0.25)",
        lambda params: SMO(params, lr=1e-3, k_ratio=0.25),
        epochs=epochs,
        device=device,
        seed=seed
    )
    
    # Run SMO (k_ratio=0.5) - middle ground
    results_swo_50 = run_experiment(
        "SMO (k_ratio=0.5)",
        lambda params: SMO(params, lr=1e-3, k_ratio=0.5),
        epochs=epochs,
        device=device,
        seed=seed
    )
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    all_results = [results_adam, results_swo, results_swo_50]
    
    for r in all_results:
        mem = r['optimizer_memory_mb']
        acc = r['final_test_acc']
        time_total = r['total_time']
        print(f"\n{r['optimizer']}:")
        print(f"  Final Test Accuracy: {acc}%")
        print(f"  Optimizer Memory:    {mem:.2f} MB")
        print(f"  Total Train Time:    {time_total:.2f}s")
    
    # Calculate savings
    baseline_mem = results_adam['optimizer_memory_mb']
    swo_mem = results_swo['optimizer_memory_mb']
    savings = (1 - swo_mem / baseline_mem) * 100 if baseline_mem > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"MEMORY SAVINGS: {savings:.1f}% reduction vs Adam")
    print(f"{'='*60}")
    
    # Accuracy gap
    baseline_acc = results_adam['final_test_acc']
    swo_acc = results_swo['final_test_acc']
    gap = baseline_acc - swo_acc
    
    print(f"ACCURACY GAP: {gap:+.2f}% (Adam: {baseline_acc}%, SMO: {swo_acc}%)")
    
    if abs(gap) < 1.0:
        verdict = "EXCELLENT: Maintains accuracy within 1%"
    elif abs(gap) < 2.0:
        verdict = "GOOD: Small accuracy trade-off"
    else:
        verdict = "NEEDS WORK: Significant accuracy loss"
    
    print(f"VERDICT: {verdict}")
    
    # Save results to benchmarks folder
    summary = {
        'adam': results_adam,
        'swo_k025': results_swo,
        'swo_k050': results_swo_50,
        'memory_savings_percent': round(savings, 1),
        'accuracy_gap_percent': round(gap, 2)
    }
    runs = [
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="Adam",
            script_name="benchmarks/suites/training/benchmark_mnist.py",
            hardware=device.upper(),
            backend=device.upper(),
            dataset="MNIST",
            model="SimpleCNN",
            seed=seed,
            batch_size=128,
            precision="fp32",
            epochs=epochs,
            metrics={
                "final_accuracy": results_adam['final_test_acc'],
                "optimizer_state_mb": results_adam['optimizer_memory_mb'],
                "total_time_s": results_adam['total_time'],
            },
            extra={"raw_result": results_adam},
        ),
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="SMO-Spatial",
            script_name="benchmarks/suites/training/benchmark_mnist.py",
            hardware=device.upper(),
            backend=device.upper(),
            dataset="MNIST",
            model="SimpleCNN",
            seed=seed,
            batch_size=128,
            precision="fp32",
            epochs=epochs,
            metrics={
                "final_accuracy": results_swo['final_test_acc'],
                "optimizer_state_mb": results_swo['optimizer_memory_mb'],
                "total_time_s": results_swo['total_time'],
            },
            extra={"raw_result": results_swo, "k_ratio": 0.25},
        ),
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="SMO-Spatial",
            script_name="benchmarks/suites/training/benchmark_mnist.py",
            hardware=device.upper(),
            backend=device.upper(),
            dataset="MNIST",
            model="SimpleCNN",
            seed=seed,
            batch_size=128,
            precision="fp32",
            epochs=epochs,
            metrics={
                "final_accuracy": results_swo_50['final_test_acc'],
                "optimizer_state_mb": results_swo_50['optimizer_memory_mb'],
                "total_time_s": results_swo_50['total_time'],
            },
            extra={"raw_result": results_swo_50, "k_ratio": 0.5},
        ),
    ]
    results_path, _ = write_benchmark_bundle(
        aggregate_filename="benchmark_results.json",
        suite_name="mnist_training_comparison",
        benchmark_family="end_to_end_training",
        summary=summary,
        runs=runs,
    )

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
