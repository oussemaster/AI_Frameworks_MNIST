# mnist_pytorch.py
import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

# 1. Transform für Normalisierung definieren
print("\n=== Transform für Normalisierung definieren ===")
transform = transforms.Compose([
    transforms.ToTensor(),  # Konvertiert zu Tensor UND normalisiert auf [0,1]
])

# 2. MNIST mit torchvision laden
print("\n=== MNIST mit torchvision laden ===")
train_dataset_torch = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset_torch = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

print("Torchvision Trainingsdaten:", len(train_dataset_torch))
print("Torchvision Testdaten:", len(test_dataset_torch))
print("Beispiel-Datensatz Shape:", train_dataset_torch[0][0].shape)

# 3. MNIST aus Hugging Face laden
print("\n=== MNIST aus Hugging Face laden ===")
dataset_hf = load_dataset('mnist')

# Definiere eine Funktion zur Normalisierung für Hugging Face
def transform_hf(example):
    image = example['image']
    # Konvertiere PIL Image zu Tensor und normalisiere
    image_tensor = transforms.ToTensor()(image)
    return {'image': image_tensor, 'label': example['label']}

# Wende die Transformation an
dataset_hf = dataset_hf.map(transform_hf)

print("Hugging Face Trainingsdaten:", len(dataset_hf['train']))
print("Hugging Face Testdaten:", len(dataset_hf['test']))

# 4. Datensatz für die Weiterverarbeitung mit PyTorch aufsetzen
print("\n=== Datensatz für PyTorch vorbereiten ===")

# DataLoader erstellen - diese kümmern sich um Batching und Shuffling
BATCH_SIZE = 32

train_loader_torch = torch.utils.data.DataLoader(
    train_dataset_torch, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

test_loader_torch = torch.utils.data.DataLoader(
    test_dataset_torch, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

# Für Hugging Face DataLoader erstellen
train_loader_hf = torch.utils.data.DataLoader(
    dataset_hf['train'], 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

test_loader_hf = torch.utils.data.DataLoader(
    dataset_hf['test'], 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

print("Torchvision DataLoader erstellt - Batch-Größe:", BATCH_SIZE)
print("Hugging Face DataLoader erstellt - Batch-Größe:", BATCH_SIZE)

# Teste einen Batch
print("\n=== Teste einen Batch ===")
for images, labels in train_loader_torch:
    print("Batch Images Shape:", images.shape)  # Sollte [32, 1, 28, 28] sein
    print("Batch Labels Shape:", labels.shape)  # Sollte [32] sein
    print("Wertebereich der Bilder: [{:.3f}, {:.3f}]".format(
        images.min().item(), images.max().item()))
    break

# Zeige ein Beispielbild an
print("\n=== Beispielbild anzeigen ===")
image, label = train_dataset_torch[0]
plt.figure(figsize=(6, 6))
plt.imshow(image.squeeze(), cmap='gray')  # squeeze() entfernt die Channel-Dimension
plt.title(f"Label: {label}")
plt.colorbar()
plt.show()

print("✅ PyTorch-Programm erfolgreich ausgeführt!")