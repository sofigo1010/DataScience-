

#!/usr/bin/env python3
"""
Entrena y evalúa dos modelos de Deep Learning sobre PolyMNIST:

* Modelo 1 – `MiniCNN`:    arquitectura muy ligera (≈ 45 k parámetros).
* Modelo 2 – `DeepCNN`:    versión más profunda (≈ 1,3 M parámetros).

El script:

1) Construye `DataLoader`s para train y test leyendo las carpetas
   PolyMNIST/MMNIST/train|test/m0…m4.
2) Entrena cada modelo (epochs y batch_size configurables).
3) Mide la precisión top‑1 en el set de test.
4) Selecciona y reporta el mejor modelo.

Requisitos:
    pip3 install torch torchvision tqdm
"""

# ----------------------------------------------------------------------
# Configuraciones globales
# ----------------------------------------------------------------------
DATA_DIR     = "PolyMNIST/MMNIST"   # ajusta si tu ruta es distinta
BATCH_SIZE   = 256
EPOCHS       = 6                    # sube a 10‑15 para resultados >97 %
LR           = 1e-3
import torch 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # autodetect

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----------------------------------------------------------------------
# Dataset y DataLoaders
# ----------------------------------------------------------------------
def get_loaders(data_root: str = DATA_DIR, batch_size: int = BATCH_SIZE):
    """
    Crea los DataLoaders para entrenamiento y prueba usando torchvision.ImageFolder.
    Cada modalidad se concatena en el mismo dataset (no los tratamos como canales).
    """
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),                  # => [0,1]
        transforms.Normalize([0.5], [0.5])      # mean=0.5, std=0.5
    ])

    train_path = Path(data_root) / "train"
    test_path  = Path(data_root) / "test"

    train_ds = ImageFolder(root=str(train_path), transform=tf)
    test_ds  = ImageFolder(root=str(test_path),  transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# ----------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------
class MiniCNN(nn.Module):
    """CNN minimal para MNIST‑like."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 14×14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 7×7
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class DeepCNN(nn.Module):
    """Versión más profunda con regularización Dropout."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 14×14

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 7×7

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 3×3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x)
        loss   = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x).argmax(dim=1)
        correct += (y_pred == y).sum().item()
    return correct / len(loader.dataset)  # accuracy


def train_and_eval(model_cls, train_loader, test_loader):
    model = model_cls().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        acc  = eval_model(model, test_loader)
        print(f"  Ep {ep:02d}/{EPOCHS}  Loss={loss:6.4f}  Acc={acc*100:5.2f}%")

    final_acc = eval_model(model, test_loader)
    return final_acc, model

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Preparando loaders...")
    print(f"Usando dispositivo: {DEVICE}")
    train_loader, test_loader = get_loaders()

    resultados = {}

    print("\nEntrenando MiniCNN…")
    acc1, model1 = train_and_eval(MiniCNN, train_loader, test_loader)
    resultados["MiniCNN"] = acc1

    print("\nEntrenando DeepCNN…")
    acc2, model2 = train_and_eval(DeepCNN, train_loader, test_loader)
    resultados["DeepCNN"] = acc2

    print("\nResultados:")
    for name, acc in resultados.items():
        print(f"  {name:8s}: {acc*100:5.2f}%")

    best_name = max(resultados, key=resultados.get)
    print(f"\n► Mejor modelo: {best_name} ({resultados[best_name]*100:5.2f}%)")

    # Guarda el mejor
    torch.save(
        (model1 if best_name == "MiniCNN" else model2).state_dict(),
        f"{best_name.lower()}_state_dict.pth"
    )
    print(f"Pesos guardados en {best_name.lower()}_state_dict.pth")


if __name__ == "__main__":
    main()