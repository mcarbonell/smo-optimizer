import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.5.0")
    .add_local_dir("smo", remote_path="/root/smo")
)

app = modal.App("smo-activation-memory-benchmark-hooks")

@app.function(image=image, gpu="A10G", timeout=600)
def run_benchmark():
    import torch
    import torch.nn as nn
    import sys
    import time
    
    sys.path.append("/root")
    
    from smo.activations_hooks import smo_squeezer

    def get_vram():
        torch.cuda.synchronize()
        # memory_allocated() shows what is currently used. 
        # For activations, we want to see the footprint during the peak of the backward pass.
        return torch.cuda.max_memory_allocated() / (1024**2)

    def create_activation_heavy_model():
        layers = []
        for _ in range(100):
            layers.append(nn.Linear(512, 512))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers).cuda()

    def benchmark(name, use_hooks=False):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = create_activation_heavy_model()
        x = torch.randn(16384, 512).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        start_time = time.time()
        
        # EL TRUCO FINAL: Envolvemos el forward Y el backward en el Squeezer
        with smo_squeezer(enabled=use_hooks):
            out = model(x)
            loss = out.sum()
            loss.backward()
        
        optimizer.step()
        torch.cuda.synchronize()
        end_time = time.time()
        
        mem = get_vram()
        print(f"📊 {name}: {mem:.2f} MB | Time: {end_time - start_time:.4f}s")
        
        del model, x, optimizer, out, loss
        torch.cuda.empty_cache()
        
        return mem

    print(f"\n🚀 Benchmarking 'SUPER MARIO HOOKS' en {torch.cuda.get_device_name(0)}...")
    print("-" * 65)
    
    m_base = benchmark("Standard (float32)")
    m_hooks = benchmark("SMO Activation Squeezer (8-bit)", use_hooks=True)

    print("\n" + "="*50)
    print("🏆 RESULTADOS FINALES DE AHORRO VRAM")
    print("="*50)
    print(f"Baseline:         {m_base:>10.2f} MB")
    print(f"SMO Squeezer:     {m_hooks:>10.2f} MB")
    print(f"Ahorro Real:      {m_base - m_hooks:>10.2f} MB ({100*(1-m_hooks/m_base):.1f}%)")
    print("="*50)
    
    if m_hooks < m_base:
        print("\n✨ ¡HITO HISTÓRICO LOGRADO! ✨")
        print("Hemos reducido el consumo de VRAM interceptando los hooks de autograd.")

if __name__ == "__main__":
    with app.run():
        run_benchmark.remote()
