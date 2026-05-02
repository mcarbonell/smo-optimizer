from smo import SMO, SMO8bit
import torch
import torch.nn as nn

def run_test(variant_name, optimizer_class):
    print(f"\n--- Testing {variant_name} ---")

    # A model with ~33 Million parameters (16.7M per layer)
    model = nn.Sequential(
        nn.Linear(4096, 4096, bias=False),
        nn.ReLU(),
        nn.Linear(4096, 4096, bias=False)
    )

    # Calculate weight memory (float32 = 4 bytes)
    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"Model Weights Memory: {weight_mem:.2f} MB")

    k_ratio = 0.25
    print(f"Initializing {variant_name} (k_ratio={k_ratio}) -> ~93.7% state compression...")
    optimizer = optimizer_class(model.parameters(), lr=1e-3, k_ratio=k_ratio)

    # Dummy forward/backward to initialize optimizer states
    inputs = torch.randn(8, 4096)
    targets = torch.randn(8, 4096)
    loss = nn.MSELoss()(model(inputs), targets)
    loss.backward()
    optimizer.step()

    # Calculate optimizer state memory
    opt_mem = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                opt_mem += v.numel() * v.element_size()

    opt_mem_mb = opt_mem / (1024 ** 2)
    standard_adam_mem = weight_mem * 2 # Standard Adam stores 2 full-sized copies (m and v)

    print(f"Standard Adam State Memory (Theoretical): {standard_adam_mem:.2f} MB")
    print(f"{variant_name} State Memory (Actual): {opt_mem_mb:.2f} MB")
    print(f"RAM Saved for optimizer states: {(1 - opt_mem_mb / standard_adam_mem) * 100:.2f}%")

def main():
    print("🎮 Super Mario Optimizer (SMO) - Memory Efficiency Demo")

    # Run test for spatial version
    run_test("SMO (Spatial)", SMO)

    # Run test for 8-bit version
    run_test("SMO8bit (Star Mode)", SMO8bit)

if __name__ == "__main__":
    main()
