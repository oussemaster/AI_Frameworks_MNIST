# mnist_pytorch_minimal.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

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
    num_workers=0,          # Paralleles Laden
    pin_memory=True,        # Schneller GPU-Transfer
    #persistent_workers=True # Vermeidet Worker-Overhead
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

# 4. Beispielbilder anzeigen
def show_sample_images(images, labels, num_images=5):
    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        # Entferne Channel Dimension und konvertiere zu NumPy
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('pytorch_mnist_samples.png', dpi=100, bbox_inches='tight')
    plt.show()

# Ersten Batch aus DataLoader holen und anzeigen
for batch_images, batch_labels in train_loader:
    print("✅ PyTorch MNIST ready!")
    print(f"Batch Shape: {batch_images.shape}")  # [64, 1, 28, 28]
    print(f"Labels: {batch_labels[:5].tolist()}")
    
    # Zeige erste 5 Bilder des Batches
    show_sample_images(batch_images, batch_labels)
    break  # Nur ersten Batch verarbeiten

print(f"Training batches: {len(train_loader)}")
print("💾 Sample images saved as 'pytorch_mnist_samples.png'")
