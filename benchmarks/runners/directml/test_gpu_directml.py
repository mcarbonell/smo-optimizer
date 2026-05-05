import os
import sys
import time

import torch

from benchmarks._paths import add_project_root_to_path

try:
    import torch_directml

    device = torch_directml.device()
    print(f"DirectML detected. Using device: {device}")
except ImportError:
    device = torch.device("cpu")
    print("torch-directml not found. Falling back to CPU.")

add_project_root_to_path()

from spectral.optim_dct_pure import SMODCTPure
from spectral.optim_walsh_pure import SMOWalshPure


def run_gpu_test(optimizer_class, name):
    print(f"\n--- Testing {name} on DirectML ---")

    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
    ).to(device)

    optimizer = optimizer_class(model.parameters(), lr=1e-3, k_ratio=0.5)

    x = torch.randn(64, 2048).to(device)
    target = torch.randn(64, 1024).to(device)
    criterion = torch.nn.MSELoss()

    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    start_time = time.time()
    steps = 20
    for i in range(steps):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print(f"  Step {i + 1}/{steps} - Loss: {loss.item():.4f}")

    total_time = time.time() - start_time
    avg_time = (total_time / steps) * 1000
    print(f"Average step time: {avg_time:.2f} ms")

    del model, optimizer, x, target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"Python: {sys.executable}")
    print(f"Torch version: {torch.__version__}")

    run_gpu_test(SMOWalshPure, "SMOWalshPure")
    run_gpu_test(SMODCTPure, "SMODCTPure")

    print("\nTest completed successfully.")
