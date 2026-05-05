import time
import modal
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from benchmarks.results_utils import make_run_record, write_benchmark_bundle

# Imagen optimizada para pruebas en Modal
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "torchvision",
        "triton>=3.0.0",
        "numpy>=1.26.0"
    )
    .add_local_dir("smo", remote_path="/root/smo")
    .add_local_dir("data", remote_path="/root/data") # if you have data
)

app = modal.App("smo-8bit-triton-cifar10")

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800, # 30 mins
)
def run_cifar10_benchmark(epochs: int = 5):
    import sys
    sys.path.append("/root")
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import triton
    
    from smo import SMO8bit, SMO8bitTriton
    
    # We define a CNN model here
    class CIFAR_CNN(nn.Module):
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

    device = 'cuda'
    
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
    
    # download to a temporary directory in the container
    data_dir = "/root/data"
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

    def run_experiment(opt_name, opt_fn):
        print(f"\n{'='*60}")
        print(f"Running: {opt_name}")
        print(f"{'='*60}")
        
        model = CIFAR_CNN().to(device)
        optimizer = opt_fn(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        results = {'epochs': []}
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            torch.cuda.synchronize() # measure accurately
            
            test_acc = evaluate(model, test_loader, device)
            epoch_time = time.time() - epoch_start
            
            results['epochs'].append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'time': epoch_time
            })
            
            print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
        total_time = time.time() - start_time
        opt_mem = get_optimizer_memory(optimizer)
        
        results['total_time'] = total_time
        results['optimizer_memory_mb'] = opt_mem
        results['final_test_acc'] = results['epochs'][-1]['test_acc']
        
        print(f"Total time: {total_time:.2f}s | Final Acc: {results['final_test_acc']:.2f}% | Mem: {opt_mem:.2f} MB")
        return results

    res_adam = run_experiment("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3))
    res_smo8_pt = run_experiment("SMO 8-bit (PyTorch)", lambda p: SMO8bit(p, lr=1e-3, k_ratio=0.25))
    res_smo8_tr = run_experiment("SMO 8-bit (Triton)", lambda p: SMO8bitTriton(p, lr=1e-3, k_ratio=0.25))

    return {
        'AdamW': res_adam,
        'SMO_8bit_PyTorch': res_smo8_pt,
        'SMO_8bit_Triton': res_smo8_tr
    }

@app.local_entrypoint()
def main():
    print("Iniciando benchmark en Modal...")
    results = run_cifar10_benchmark.remote(epochs=5)
    
    print("\n" + "="*60)
    print("🏆 RESULTADOS FINALES DE ENTRENAMIENTO CIFAR-10 (5 Epochs)")
    print("="*60)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy Final: {res['final_test_acc']:.2f}%")
        print(f"  Tiempo Total:   {res['total_time']:.2f} s")
        print(f"  Memoria Optim:  {res['optimizer_memory_mb']:.2f} MB")
    
    runs = [
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="AdamW",
            script_name="benchmarks/runners/modal/benchmark_8bit_triton_cifar10_modal.py",
            hardware="NVIDIA A10G",
            backend="CUDA",
            dataset="CIFAR-10",
            model="CIFAR_CNN",
            batch_size=128,
            precision="fp32",
            epochs=5,
            metrics={
                "final_accuracy": results["AdamW"]["final_test_acc"],
                "optimizer_state_mb": results["AdamW"]["optimizer_memory_mb"],
                "total_time_s": results["AdamW"]["total_time"],
            },
            extra={"raw_result": results["AdamW"]},
        ),
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="SMO-Spatial-8bit",
            script_name="benchmarks/runners/modal/benchmark_8bit_triton_cifar10_modal.py",
            hardware="NVIDIA A10G",
            backend="CUDA",
            dataset="CIFAR-10",
            model="CIFAR_CNN",
            batch_size=128,
            precision="fp32",
            epochs=5,
            metrics={
                "final_accuracy": results["SMO_8bit_PyTorch"]["final_test_acc"],
                "optimizer_state_mb": results["SMO_8bit_PyTorch"]["optimizer_memory_mb"],
                "total_time_s": results["SMO_8bit_PyTorch"]["total_time"],
            },
            extra={"raw_result": results["SMO_8bit_PyTorch"], "k_ratio": 0.25, "implementation": "PyTorch"},
        ),
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="SMO-Spatial-8bit-Triton",
            script_name="benchmarks/runners/modal/benchmark_8bit_triton_cifar10_modal.py",
            hardware="NVIDIA A10G",
            backend="CUDA",
            dataset="CIFAR-10",
            model="CIFAR_CNN",
            batch_size=128,
            precision="fp32",
            epochs=5,
            metrics={
                "final_accuracy": results["SMO_8bit_Triton"]["final_test_acc"],
                "optimizer_state_mb": results["SMO_8bit_Triton"]["optimizer_memory_mb"],
                "total_time_s": results["SMO_8bit_Triton"]["total_time"],
            },
            extra={"raw_result": results["SMO_8bit_Triton"], "k_ratio": 0.25, "implementation": "Triton"},
        ),
    ]
    results_path, _ = write_benchmark_bundle(
        aggregate_filename="benchmark_8bit_triton_cifar10_results.json",
        suite_name="cifar10_modal_8bit_triton",
        benchmark_family="end_to_end_training",
        summary=results,
        runs=runs,
    )
    print(f"\nResultados guardados en {results_path}")

if __name__ == "__main__":
    main()
