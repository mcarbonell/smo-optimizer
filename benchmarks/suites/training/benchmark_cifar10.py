#!/usr/bin/env python3
"""
Benchmark: SMO (SWO) vs Standard Adam on CIFAR-10.
A more challenging test than MNIST for validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
# Benchmark classification: family=end_to_end_training, category=end_to_end, status=canonical
from smo import SMO
from benchmarks._paths import DATA_DIR
from benchmarks.results_utils import make_run_record, write_benchmark_bundle


class CIFAR_CNN(nn.Module):
    """A moderate CNN for CIFAR-10. ~1.2M parameters."""
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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


def run_experiment(optimizer_name, optimizer_fn, epochs=5, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Running: {optimizer_name}")
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
    
    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    model = CIFAR_CNN().to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    results = {
        'optimizer': optimizer_name,
        'parameters': param_count,
        'epochs': [],
        'train_time': 0
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
    opt_mem = get_optimizer_memory(optimizer)
    
    results['total_time'] = round(total_time, 2)
    results['optimizer_memory_mb'] = round(opt_mem, 2)
    results['final_test_acc'] = results['epochs'][-1]['test_acc']
    
    print(f"\nFinal Results:")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Final test accuracy: {results['final_test_acc']:.2f}%")
    print(f"  Optimizer state memory: {opt_mem:.2f} MB")
    
    return results


def main():
    device = 'cpu'
    epochs = 5
    
    print("="*60)
    print("SWO Benchmark: SMO vs Standard Adam on CIFAR-10")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print("WARNING: On CPU this takes ~60-90 minutes total. Be patient.")
    
    results_adam = run_experiment(
        "Standard Adam",
        lambda params: torch.optim.Adam(params, lr=1e-3),
        epochs=epochs,
        device=device
    )
    
    results_swo = run_experiment(
        "SMO (k_ratio=0.25)",
        lambda params: SMO(params, lr=1e-3, k_ratio=0.25),
        epochs=epochs,
        device=device
    )
    
    results_swo_50 = run_experiment(
        "SMO (k_ratio=0.5)",
        lambda params: SMO(params, lr=1e-3, k_ratio=0.5),
        epochs=epochs,
        device=device
    )
    
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
    
    baseline_mem = results_adam['optimizer_memory_mb']
    swo_mem = results_swo['optimizer_memory_mb']
    savings = (1 - swo_mem / baseline_mem) * 100 if baseline_mem > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"MEMORY SAVINGS: {savings:.1f}% reduction vs Adam")
    print(f"{'='*60}")
    
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
            script_name="benchmarks/suites/training/benchmark_cifar10.py",
            hardware=device.upper(),
            backend="CPU",
            dataset="CIFAR-10",
            model="CIFAR_CNN",
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
            script_name="benchmarks/suites/training/benchmark_cifar10.py",
            hardware=device.upper(),
            backend="CPU",
            dataset="CIFAR-10",
            model="CIFAR_CNN",
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
            script_name="benchmarks/suites/training/benchmark_cifar10.py",
            hardware=device.upper(),
            backend="CPU",
            dataset="CIFAR-10",
            model="CIFAR_CNN",
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
        aggregate_filename="benchmark_cifar10_results.json",
        suite_name="cifar10_training_comparison",
        benchmark_family="end_to_end_training",
        summary=summary,
        runs=runs,
    )

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
