import os
import modal
import sys

# Definimos una imagen con PyTorch y Triton instalados
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.5.0", "triton>=3.0.0")
)

app = modal.App("smo-triton-test")

@app.function(
    image=image,
    gpu="T4", # Una T4 barata es perfecta para testear si hay CUDA
    timeout=300,
)
def check_environment():
    import torch
    import triton
    
    print("\n--- Modal GPU & Triton Check ---")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detectada: {gpu_name}")
        print(f"✅ PyTorch versión: {torch.__version__}")
        print(f"✅ Triton versión: {triton.__version__}")
        return True
    else:
        print("❌ No se detectó GPU CUDA.")
        return False

@app.local_entrypoint()
def main():
    print("Iniciando conexión con Modal...")
    success = check_environment.remote()
    if success:
        print("\n🚀 ¡Conexión con Modal perfecta! Listo para compilar kernels Triton.")
    else:
        print("\n⚠️ Hubo un problema detectando la GPU en Modal.")
