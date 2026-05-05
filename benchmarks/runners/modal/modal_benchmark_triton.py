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

app = modal.App("smo-triton-benchmark")

@app.function(
    image=image,
    gpu="T4", # Usamos T4. Si falla por falta de memoria, cambia a A10G.
    timeout=600,
)
def run_triton_benchmark():
    import sys
    sys.path.append("/root")
    
    import torch
    import triton
    
    # Importamos nuestros optimizadores
    from smo import SMO
    from smo.optim_triton import SMOTriton, HAS_TRITON
    
    if not HAS_TRITON:
        print("❌ Triton no está instalado correctamente en la imagen de Modal.")
        return
        
    print("\n" + "="*60)
    print(f"🚀 Iniciando Benchmark de Fused Kernel en: {torch.cuda.get_device_name(0)}")
    print("="*60)

    # 1. Configuración del Test (Capa densa gigante para simular un LLM)
    # Tamaño de la matriz de pesos: 8192 x 8192 (~67 Millones de parámetros)
    # Esto consumirá mucha VRAM, ideal para notar el cuello de botella.
    SIZE = 8192
    device = "cuda"
    dtype = torch.float32 # Usamos FP32 para que el movimiento de memoria sea evidente

    print(f"Creando capa de prueba de {SIZE}x{SIZE} ({SIZE * SIZE / 1e6:.1f}M parámetros)")
    
    def benchmark_optimizer(opt_name, optimizer_fn, iters=100):
        # Creamos tensores frescos para no compartir estado
        layer = torch.nn.Linear(SIZE, SIZE, bias=False).to(device, dtype=dtype)
        optimizer = optimizer_fn(layer.parameters())
        
        # Dummy Input
        x = torch.randn(128, SIZE, device=device, dtype=dtype)
        
        # Warmup (Importante para compilar los kernels de Triton la primera vez)
        print(f"  [Warmup] Compilando y calentando {opt_name}...")
        for _ in range(5):
            y = layer(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        
        # Benchmark real
        print(f"  [Running] Midiendo {iters} iteraciones...")
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

    # 2. Ejecutar Pruebas
    iters = 50
    
    # Prueba A: Standard AdamW
    ms_adam = benchmark_optimizer(
        "Standard AdamW",
        lambda params: torch.optim.AdamW(params, lr=1e-3),
        iters=iters
    )
    
    # Prueba B: SMO PyTorch (Spatial con Pooling)
    ms_smo_pt = benchmark_optimizer(
        "SMO (PyTorch Spatial k=0.5)",
        lambda params: SMO(params, lr=1e-3, k_ratio=0.5),
        iters=iters
    )
    
    # Prueba C: SMO Triton (Fused Kernel)
    ms_smo_triton = benchmark_optimizer(
        "SMO (Triton Fused Kernel k=0.5)",
        lambda params: SMOTriton(params, lr=1e-3, k_ratio=0.5),
        iters=iters
    )
    
    # 3. Resultados Finales
    print("\n" + "="*60)
    print("🏆 RESULTADOS DE VELOCIDAD (GPU VRAM Bandwidth)")
    print("="*60)
    print(f"Standard AdamW:           {ms_adam:.2f} ms/step")
    print(f"SMO (PyTorch Native):     {ms_smo_pt:.2f} ms/step")
    print(f"SMO (Triton Fused):       {ms_smo_triton:.2f} ms/step")
    print("-" * 60)
    
    speedup_adam = ms_adam / ms_smo_triton
    speedup_pt = ms_smo_pt / ms_smo_triton
    
    print(f"⚡ Triton Kernel es {speedup_adam:.2f}x más rápido que AdamW")
    print(f"⚡ Triton Kernel es {speedup_pt:.2f}x más rápido que SMO en PyTorch")
    
    if speedup_adam > 1.0:
        print("\n🎉 ¡ÉXITO! El Fused Kernel ha roto el Memory Wall. Es más rápido que Adam.")
    else:
        print("\n⚠️ Triton fue más lento. Necesitamos ajustar el BLOCK_SIZE o tunear la caché L2.")

@app.local_entrypoint()
def main():
    print("Enviando código a Modal y solicitando GPU NVIDIA...")
    run_triton_benchmark.remote()

