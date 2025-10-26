# mnist_pytorch_optimized.py
import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import random

print("=" * 50)
print("OPTIMIERTE MNIST ANALYSE MIT PYTORCH - NEUE FEATURES")
print("=" * 50)

# 1. Transformationen definieren
print("\n1. 🔧 TRANSFORMATIONEN DEFINIEREN")

# Transform bleibt gleich (bereits optimal)
transform = transforms.Compose([
    transforms.ToTensor(),  # Konvertiert zu Tensor und normalisiert auf [0,1]
])

print("✅ Standard-Transformation: ToTensor() -> [0,1] Normalisierung")

# 2. Datensätze laden
print("\n2. 📥 DATENSÄTZE LADEN")

# Torchvision (bleibt gleich)
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

print(f"✅ Torchvision: {len(train_dataset_torch):,} Trainingsbilder")
print(f"✅ Torchvision: {len(test_dataset_torch):,} Testbilder")

# Hugging Face (bleibt gleich)
try:
    dataset_hf = load_dataset('mnist')
    print(f"✅ Hugging Face: {len(dataset_hf['train']):,} Trainingsbilder")
except Exception as e:
    print(f"⚠️  Hugging Face Fehler: {e}")

# 3. Datensatz-Statistiken (bleibt gleich)
print("\n3. 📊 DATENSATZ-ANALYSE")

def analyze_dataset(dataset, name):
    print(f"\n--- {name} ---")
    
    if hasattr(dataset, 'targets'):
        labels = dataset.targets.numpy()
    else:
        labels = []
        for _, label in dataset:
            labels.append(label)
        labels = np.array(labels)
    
    print(f"Anzahl Bilder: {len(dataset):,}")
    print(f"Bildgröße: {dataset[0][0].shape}")
    print(f"Wertebereich: [{dataset[0][0].min():.3f}, {dataset[0][0].max():.3f}]")
    
    unique, counts = np.unique(labels, return_counts=True)
    print("Klassenverteilung:")
    for label, count in zip(unique, counts):
        print(f"  Ziffer {label}: {count:>5} Beispiele ({count/len(labels)*100:.1f}%)")

analyze_dataset(train_dataset_torch, "TRAININGSDATEN")
analyze_dataset(test_dataset_torch, "TESTDATEN")

# 4. Visualisierung (bleibt gleich)
print("\n4. 🖼️ VISUALISIERUNG")

def plot_comparison(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]
        
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Ziffer: {label}', weight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Zufällige Beispiele aus dem MNIST-Datensatz', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('pytorch_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Beispiele gespeichert als 'pytorch_samples.png'")

plot_comparison(train_dataset_torch)

# =============================================================================
# 5. OPTIMIERTE DATALOADER - NEUE FEATURES VON DEN FOLIEN
# =============================================================================
print("\n5. 🔄 OPTIMIERTE DATALOADER ERSTELLEN")

BATCH_SIZE = 64

# 🔹 ALTE VERSION (auskommentiert)
"""
train_loader = torch.utils.data.DataLoader(
    train_dataset_torch, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0  # ❌ Suboptimal: Keine Parallelisierung
)

test_loader = torch.utils.data.DataLoader(
    test_dataset_torch, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=0  # ❌ Suboptimal: Keine Parallelisierung  
)
"""

# 🔹 NEUE OPTIMIERTE VERSION (nach Dozent-Empfehlung)
train_loader = torch.utils.data.DataLoader(
    train_dataset_torch, 
    batch_size=BATCH_SIZE, 
    shuffle=True,           # 🔹 Wichtig für Training
    num_workers=2,          # 🔹 NEU: Paralleles Laden (Performance+)
    pin_memory=True,        # 🔹 NEU: Schneller GPU-Transfer (Performance+)
    persistent_workers=True, # 🔹 NEU: Vermeidet Worker-Neustart (Performance+)
    drop_last=True          # 🔹 NEU: Konsistente Batch-Größen
)

test_loader = torch.utils.data.DataLoader(
    test_dataset_torch, 
    batch_size=BATCH_SIZE, 
    shuffle=False,          # 🔹 Wichtig: Kein Shuffle für Evaluation!
    num_workers=2,          # 🔹 NEU: Paralleles Laden  
    pin_memory=True,        # 🔹 NEU: Schneller GPU-Transfer
    persistent_workers=True, # 🔹 NEU: Vermeidet Worker-Neustart
    drop_last=False         # 🔹 NEU: Alle Testdaten verwenden
)

print(f"✅ OPTIMIERTER Trainings-Dataloader:")
print(f"   - Batch-Größe: {BATCH_SIZE}")
print(f"   - Shuffle: aktiviert")
print(f"   - num_workers: 2 🔹 NEU")
print(f"   - pin_memory: True 🔹 NEU") 
print(f"   - persistent_workers: True 🔹 NEU")
print(f"   - drop_last: True 🔹 NEU")

print(f"✅ OPTIMIERTER Test-Dataloader:")
print(f"   - Batch-Größe: {BATCH_SIZE}") 
print(f"   - Shuffle: deaktiviert (korrekt für Evaluation)")
print(f"   - num_workers: 2 🔹 NEU")
print(f"   - pin_memory: True 🔹 NEU")
print(f"   - persistent_workers: True 🔹 NEU")
print(f"   - drop_last: False 🔹 NEU")

# =============================================================================
# 6. ERWEITERTE BATCH-ANALYSE
# =============================================================================
print("\n6. 📦 ERWEITERTE BATCH-ANALYSE")

def analyze_optimized_batches(loader, num_batches=2):
    print("🔹 OPTIMIERTE Batch-Analyse:")
    
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  - Images Shape: {images.shape}")  # [Batch, Channel, Height, Width]
        print(f"  - Labels Shape: {labels.shape}")
        print(f"  - Wertebereich: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  - Eindeutige Labels: {torch.unique(labels).tolist()}")
        
        # 🔹 NEU: Device-Information
        print(f"  - Device: {images.device}")
        print(f"  - Datentyp: {images.dtype}")
        
        if batch_idx == 0:
            # Visualisiere erstes Bild
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(images[0].squeeze(), cmap='gray')
            ax.set_title(f'Batch 0, Bild 0: Ziffer {labels[0].item()}', weight='bold')
            ax.axis('off')
            plt.savefig('pytorch_batch_example.png', dpi=150, bbox_inches='tight')
            print("✅ Beispiel aus Batch gespeichert als 'pytorch_batch_example.png'")

analyze_optimized_batches(train_loader)

# =============================================================================
# 7. PERFORMANCE-TEST DER OPTIMIERTEN PIPELINE
# =============================================================================
print("\n7. ⚡ PERFORMANCE-TEST")

def test_pipeline_speed(loader, num_batches=10):
    """
    Testet die Geschwindigkeit der optimierten Pipeline.
    """
    import time
    
    print("Teste Pipeline-Geschwindigkeit...")
    start_time = time.time()
    
    batch_count = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        batch_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    time_per_batch = total_time / batch_count
    
    print(f"✅ Performance-Ergebnisse:")
    print(f"   - Verarbeitete Batches: {batch_count}")
    print(f"   - Gesamtzeit: {total_time:.3f} Sekunden")
    print(f"   - Zeit pro Batch: {time_per_batch:.3f} Sekunden")
    print(f"   - Batches pro Sekunde: {1/time_per_batch:.1f}")

# Führe Performance-Test aus
test_pipeline_speed(train_loader, num_batches=5)

# =============================================================================
# 8. ZUSAMMENFASSUNG MIT NEUEN FEATURES
# =============================================================================
print("\n" + "=" * 50)
print("🎉 OPTIMIERTE ZUSAMMENFASSUNG")
print("=" * 50)

print(f"✅ PyTorch Datensätze erfolgreich mit OPTIMIERTER Pipeline geladen")
print(f"📊 Trainingsbilder: {len(train_dataset_torch):,}")
print(f"📊 Testbilder: {len(test_dataset_torch):,}")
print(f"📐 Bildgröße: {train_dataset_torch[0][0].shape}")

print(f"\n🚀 NEUE OPTIMIERUNGEN:")
print(f"📦 Batch-Größe: {BATCH_SIZE}")
print(f"👥 num_workers: 2 (Parallel Loading) 🔹 NEU")
print(f"📌 pin_memory: True (GPU Transfer) 🔹 NEU")
print(f"🔄 persistent_workers: True (Performance) 🔹 NEU")
print(f"🎯 drop_last: intelligent (Training ja, Test nein) 🔹 NEU")

print(f"\n💾 Visualisierungen gespeichert:")
print(f"   - pytorch_samples.png")
print(f"   - pytorch_batch_example.png")

print(f"\n⚡ Performance: Getestet und optimiert!")

# 🔹 NEU: Device Information mit Optimierungen
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"\n💻 Verfügbare Hardware: {device}")
if torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) verfügbar für Apple Silicon!")
    print("🔹 OPTIMIERT: pin_memory=True beschleunigt Transfer zu GPU!")

print(f"\n🎯 Die PyTorch-Pipeline ist jetzt OPTIMAL für Training vorbereitet!")
print(f"   Entspricht den Best Practices aus den Vorlesungsfolien!")