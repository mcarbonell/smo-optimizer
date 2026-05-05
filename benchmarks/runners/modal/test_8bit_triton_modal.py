import os
import time
import modal

# Imagen optimizada para pruebas de Triton en Modal
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "triton>=3.0.0",
        "numpy>=1.26.0"
    )
    # Montamos todo el directorio actual para que tenga acceso al paquete smo
    .add_local_dir("smo", remote_path="/root/smo")
)

app = modal.App("smo-8bit-triton-benchmark")

@app.function(
    image=image,
    gpu="A10G", # Usamos A10G para tener suficiente VRAM y potencia
    timeout=600,
)
def run_8bit_triton_benchmark():
    import sys
    sys.path.append("/root")
    
    import torch
    import triton
    
    # Importamos nuestros optimizadores
    from smo import SMO8bit, SMO8bitTriton
    
    print("\n" + "="*60)
    print(f"🚀 Iniciando Benchmark de SMO 8-bit en: {torch.cuda.get_device_name(0)}")
    print("="*60)

    # Configuración del Test
    SIZE = 4096 # Tamaño manejable pero representativo
    device = "cuda"
    dtype = torch.float32

    print(f"Creando capa de prueba de {SIZE}x{SIZE} ({SIZE * SIZE / 1e6:.1f}M parámetros)")
    
    def benchmark_optimizer(opt_name, optimizer_fn, iters=50):
        layer = torch.nn.Linear(SIZE, SIZE, bias=False).to(device, dtype=dtype)
        optimizer = optimizer_fn(layer.parameters())
        
        # Dummy Input
        x = torch.randn(128, SIZE, device=device, dtype=dtype)
        
        # Warmup
        print(f"  [Warmup] {opt_name}...")
        for _ in range(5):
            y = layer(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"  [Running] {iters} iteraciones...")
        start_time = time.time()
        
        for _ in range(iters):
            y = layer(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        ms_per_step = (total_time / iters) * 1000
        
        print(f"  ✅ {opt_name}: {total_time:.2f}s total | {ms_per_step:.2f} ms/step")
        return ms_per_step

    iters = 50
    
    # 1. AdamW (Referencia)
    ms_adam = benchmark_optimizer(
        "Standard AdamW",
        lambda params: torch.optim.AdamW(params, lr=1e-3),
        iters=iters
    )
    
    # 2. SMO 8-bit PyTorch
    ms_smo8_pt = benchmark_optimizer(
        "SMO 8-bit (PyTorch Native)",
        lambda params: SMO8bit(params, lr=1e-3, k_ratio=0.25, block_size=64),
        iters=iters
    )
    
    # 3. SMO 8-bit Triton
    ms_smo8_triton = benchmark_optimizer(
        "SMO 8-bit (Triton Fused)",
        lambda params: SMO8bitTriton(params, lr=1e-3, k_ratio=0.25, block_size=64),
        iters=iters
    )
    
    # Resultados Finales
    print("\n" + "="*60)
    print("🏆 RESULTADOS FINALES")
    print("="*60)
    print(f"Standard AdamW:           {ms_adam:.2f} ms/step")
    print(f"SMO 8-bit (PyTorch):      {ms_smo8_pt:.2f} ms/step")
    print(f"SMO 8-bit (Triton):       {ms_smo8_triton:.2f} ms/step")
    print("-" * 60)
    
    speedup = ms_smo8_pt / ms_smo8_triton
    print(f"⚡ Triton es {speedup:.2f}x más rápido que la versión PyTorch")
    
    # Verificación rápida de que los pesos cambian
    layer = torch.nn.Linear(8, 8).to(device)
    initial_weight = layer.weight.clone()
    optimizer = SMO8bitTriton(layer.parameters(), lr=1.0)
    x = torch.randn(1, 8, device=device)
    layer(x).sum().backward()
    optimizer.step()
    
    diff = (layer.weight - initial_weight).abs().sum().item()
    if diff > 0:
        print("\n✅ Verificación funcional: Los pesos se han actualizado correctamente.")
    else:
        print("\n❌ Error funcional: Los pesos no han cambiado tras el step.")

@app.local_entrypoint()
def main():
    run_8bit_triton_benchmark.remote()
