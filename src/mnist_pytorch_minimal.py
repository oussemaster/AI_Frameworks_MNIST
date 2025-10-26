# mnist_pytorch_minimal.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Transform für Normalisierung
transform = transforms.Compose([
    transforms.ToTensor()  # Konvertiert zu Tensor + normalisiert auf [0,1]
])

# 2. Von torchvision laden
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. DataLoader mit Optimierungen
train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,           # Shuffling für Training
    num_workers=2,          # Paralleles Laden
    pin_memory=True,        # Schneller GPU-Transfer
    persistent_workers=True # Vermeidet Worker-Overhead
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,          # Kein Shuffling für Test
    num_workers=2,
    pin_memory=True
)

print("✅ PyTorch MNIST ready!")
print(f"Training batches: {len(train_loader)}")