import torch
import time
import sys
import os

# Intentar importar torch-directml para aceleración en AMD
try:
    import torch_directml
    device = torch_directml.device()
    print(f"✅ DirectML detectado. Usando dispositivo: {device}")
except ImportError:
    device = torch.device("cpu")
    print("❌ torch-directml no encontrado. Usando CPU (esto será lento).")

# Asegurar que el path del proyecto esté disponible
sys.path.append(os.getcwd())

from spectral.optim_walsh_pure import SMOWalshPure
from spectral.optim_dct_pure import SMODCTPure

def run_gpu_test(optimizer_class, name):
    print(f"\n--- Probando {name} en GPU (DirectML) ---")
    
    # Crear un modelo simple y datos sintéticos
    # Capas grandes para que se note la optimización espectral
    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024)
    ).to(device)
    
    optimizer = optimizer_class(model.parameters(), lr=1e-3, k_ratio=0.5)
    
    # Dummy data
    x = torch.randn(64, 2048).to(device)
    target = torch.randn(64, 1024).to(device)
    criterion = torch.nn.MSELoss()
    
    # Calentamiento (Warmup)
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Benchmark de tiempo
    start_time = time.time()
    steps = 20
    for i in range(steps):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"  Paso {i+1}/{steps} - Loss: {loss.item():.4f}")
            
    total_time = time.time() - start_time
    avg_time = (total_time / steps) * 1000
    print(f"⏱️ Tiempo promedio por paso: {avg_time:.2f} ms")
    
    # Limpiar memoria
    del model, optimizer, x, target
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f"Python: {sys.executable}")
    print(f"Torch version: {torch.__version__}")
    
    run_gpu_test(SMOWalshPure, "SMOWalshPure (Optimizado)")
    run_gpu_test(SMODCTPure, "SMODCTPure (Optimizado)")
    
    print("\n✅ Prueba finalizada con éxito.")
