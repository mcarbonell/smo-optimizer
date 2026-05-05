# Benchmark classification: family=end_to_end_training, category=end_to_end, status=canonical
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from benchmarks._paths import DATA_DIR, add_project_root_to_path

try:
    import torch_directml

    device = torch_directml.device()
    print(f"Using DirectML on: {device}")
except ImportError:
    device = torch.device("cpu")
    print("DirectML unavailable, using CPU.")

add_project_root_to_path()

from smo.optim import SMO
from spectral.optim_dct_pure import SMODCTPure
from spectral.optim_walsh_pure import SMOWalshPure


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


def train_epoch(model, loader, optimizer, criterion):
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


def evaluate(model, loader):
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


def run_experiment(name, optimizer_fn, epochs=3):
    print(f"\nStarting: {name}")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = CIFAR_CNN().to(device)
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)
        print(f"  Ep {epoch} | Loss: {loss:.4f} | TrAcc: {train_acc:.2f}% | TeAcc: {test_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    return {"acc": test_acc, "time": total_time}


if __name__ == "__main__":
    results = {}
    epochs = 3

    results["AdamW"] = run_experiment("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3), epochs)
    results["SMOWalshPure"] = run_experiment("SMOWalshPure", lambda p: SMOWalshPure(p, lr=1e-3, k_ratio=0.5), epochs)
    results["SMODCTPure"] = run_experiment("SMODCTPure", lambda p: SMODCTPure(p, lr=1e-3, k_ratio=0.5), epochs)

    print("\n" + "=" * 60)
    print("FINAL DIRECTML GPU SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        print(f"{name:15} | Acc: {res['acc']:.2f}% | Time: {res['time']:>7.2f}s")
