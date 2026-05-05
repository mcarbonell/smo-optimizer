"""
benchmarks/benchmark_activations.py — Memory Benchmark for 8-bit Activations

Compares a standard model vs a model with 8-bit quantized activations.
"""

# Benchmark classification: family=activation_memory, category=microbenchmark, status=canonical
import torch
import torch.nn as nn
import time
import os
import sys

from benchmarks._paths import add_project_root_to_path

add_project_root_to_path()

from smo.activations_8bit import wrap_model_activations

def measure_memory(model, input_size, iterations=5):
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = torch.randn(input_size).to(device)
    targets = torch.randn(input_size[0], 1024).to(device) # Assuming output dim is 1024
    
    # Warmup
    for _ in range(2):
        output = model(inputs)
        loss = F_mse(output, targets)
        loss.backward()
        model.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_mem = 0
    if torch.cuda.is_available():
        start_mem = torch.cuda.memory_allocated()

    start_time = time.time()
    for _ in range(iterations):
        output = model(inputs)
        loss = F_mse(output, targets)
        loss.backward()
        model.zero_grad()
    
    end_time = time.time()
    
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() - start_mem
    
    return {
        "peak_mem_mb": peak_mem / (1024**2),
        "avg_time": (end_time - start_time) / iterations
    }

def F_mse(output, target):
    return ((output - target)**2).mean()

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A deep model to accumulate activations
        layers = []
        for _ in range(10):
            layers.append(nn.Linear(4096, 4096))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(4096, 1024)

    def forward(self, x):
        return self.head(self.net(x))

def main():
    print("🧪 SMO Activation Optimization Prototype - Benchmark")
    print("-" * 50)
    
    batch_size = 32
    input_size = (batch_size, 4096)
    
    # 1. Baseline
    print("Running Baseline (Standard float32 activations)...")
    model_base = LargeModel()
    res_base = measure_memory(model_base, input_size)
    
    # 2. SMO 8-bit
    print("Running SMO 8-bit Activations...")
    model_smo = LargeModel()
    model_smo = wrap_model_activations(model_smo, block_size=64)
    res_smo = measure_memory(model_smo, input_size)
    
    print("\n" + "="*30)
    print("RESULTS COMPARISON")
    print("="*30)
    print(f"{'Metric':<20} | {'Baseline':<12} | {'SMO 8-bit':<12} | {'Savings/Diff'}")
    print("-" * 65)
    
    mem_diff = res_base['peak_mem_mb'] - res_smo['peak_mem_mb']
    mem_perc = (1 - res_smo['peak_mem_mb'] / res_base['peak_mem_mb']) * 100 if res_base['peak_mem_mb'] > 0 else 0
    
    print(f"{'Peak Activation RAM':<20} | {res_base['peak_mem_mb']:>9.2f} MB | {res_smo['peak_mem_mb']:>9.2f} MB | {mem_perc:>6.1f}%")
    print(f"{'Avg Step Time':<20} | {res_base['avg_time']:>9.4f} s  | {res_smo['avg_time']:>9.4f} s  | {res_smo['avg_time']/res_base['avg_time']:>6.2f}x")

    print("\nNote: On CPU, 'Peak RAM' might show 0 as max_memory_allocated is CUDA-only.")
    print("If running on CPU, the benchmark mainly measures the timing overhead.")

if __name__ == "__main__":
    main()
