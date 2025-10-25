# mnist_tensorflow_optimized.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import random
import os

print("=" * 50)
print("OPTIMIERTE MNIST DATENSATZ ANALYSE MIT NEUEN FEATURES")
print("=" * 50)

# 1. MNIST aus beiden Quellen laden
print("\n1. 📥 DATENSÄTZE LADEN")

# Aus Keras (bleibt gleich - bewährt und gut)
(x_train_keras, y_train_keras), (x_test_keras, y_test_keras) = mnist.load_data()
print(f"✅ Keras: {x_train_keras.shape[0]:,} Trainingsbilder, {x_test_keras.shape[0]:,} Testbilder")

# Aus TFDS (bleibt gleich)
ds_train_tfds, ds_test_tfds = tfds.load('mnist', split=['train', 'test'], 
                                       as_supervised=True, shuffle_files=True)
print(f"✅ TFDS: Datensätze erfolgreich geladen")

# 2. Normalisierung
print("\n2. 🔧 DATENVORBEREITUNG")

def normalize_data(images):
    """Normalisiert Bilder auf [0,1] Bereich"""
    return images.astype('float32') / 255.0

x_train_norm = normalize_data(x_train_keras)
x_test_norm = normalize_data(x_test_keras)

print(f"📊 Wertebereich vor Normalisierung: [{x_train_keras.min()}, {x_train_keras.max()}]")
print(f"📊 Wertebereich nach Normalisierung: [{x_train_norm.min():.3f}, {x_train_norm.max():.3f}]")

# 3. Datensatz-Statistiken anzeigen (bleibt gleich)
print("\n3. 📊 DATENSATZ-STATISTIKEN")

def print_dataset_stats(images, labels, name):
    print(f"\n--- {name} ---")
    print(f"Shape: {images.shape}")
    print(f"Datentyp: {images.dtype}")
    print(f"Eindeutige Klassen: {np.unique(labels)}")
    
    unique, counts = np.unique(labels, return_counts=True)
    print("Klassenverteilung:")
    for label, count in zip(unique, counts):
        print(f"  Ziffer {label}: {count:>5} Beispiele ({count/len(labels)*100:.1f}%)")

print_dataset_stats(x_train_norm, y_train_keras, "TRAININGSDATEN")
print_dataset_stats(x_test_norm, y_test_keras, "TESTDATEN")

# 4. Zufällige Beispiele anzeigen (bleibt gleich)
print("\n4. 🖼️ VISUALISIERUNG ZUFÄLLIGER BILDER")

def plot_random_samples(images, labels, num_samples=10):
    indices = random.sample(range(len(images)), num_samples)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx], cmap='gray')
        axes[i].set_title(f'Label: {labels[idx]}', fontsize=12, weight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_random_samples.png', dpi=150, bbox_inches='tight')
    print(f"✅ Zufällige Beispiele gespeichert als 'mnist_random_samples.png'")

plot_random_samples(x_train_norm, y_train_keras)

# 5. Erste Bild von jeder Klasse anzeigen (bleibt gleich)
print("\n5. 🔢 BEISPIEL FÜR JEDE ZIFFER")

def plot_one_per_class(images, labels):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for digit in range(10):
        idx = np.where(labels == digit)[0][0]
        
        axes[digit].imshow(images[idx], cmap='viridis')
        axes[digit].set_title(f'Ziffer: {digit}', fontsize=14, weight='bold', pad=10)
        axes[digit].axis('off')
        
        height, width = images[idx].shape
        for i in range(height):
            for j in range(width):
                if images[idx][i, j] > 0.5:
                    axes[digit].text(j, i, f'{images[idx][i, j]:.1f}', 
                                   ha='center', va='center', fontsize=6, 
                                   color='white', alpha=0.7)
    
    plt.suptitle('Erstes Beispiel für jede Ziffer 0-9 mit Pixelwerten', 
                 fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('mnist_per_class.png', dpi=150, bbox_inches='tight')
    print("✅ Beispiele pro Klasse gespeichert als 'mnist_per_class.png'")

plot_one_per_class(x_train_norm, y_train_keras)

# =============================================================================
# 6. OPTIMIERTE DATENSATZ-VORBEREITUNG - NEUE FEATURES VON DEN FOLIEN
# =============================================================================
print("\n6. 🎯 OPTIMIERTE DATENSATZ-VORBEREITUNG")

# 🔹 ALTE VERSION (auskommentiert)
"""
def create_tf_dataset(images, labels, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
"""

# 🔹 NEUE OPTIMIERTE VERSION (nach Dozent-Empfehlung)
def create_optimized_tf_dataset(images, labels, batch_size=32, shuffle=True, 
                               training=True, cache_data=True):
    """
    Optimierte Version mit Cache und verbessertem Prefetching.
    Entspricht den Empfehlungen aus den neuen Folien.
    """
    # 1. Dataset aus Tensoren erstellen
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # 2. Normalisierung anwenden
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    # 🔹 NEU: CACHE für Performance (Dozent-Empfehlung)
    if cache_data:
        dataset = dataset.cache()  # Normalisierte Daten im RAM cachen
    
    # 3. Shuffling (nur für Training)
    if shuffle and training:
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    
    # 4. Batching
    dataset = dataset.batch(batch_size)
    
    # 🔹 NEU: VERBESSERTES PREFETCHING (Dozent-Empfehlung)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Parameter
BATCH_SIZE = 64

# 🔹 ALTE VERSION (auskommentiert)
"""
train_dataset = create_tf_dataset(x_train_norm, y_train_keras, 
                                 batch_size=BATCH_SIZE, shuffle=True)
test_dataset = create_tf_dataset(x_test_norm, y_test_keras, 
                                batch_size=BATCH_SIZE, shuffle=False)
"""

# 🔹 NEUE OPTIMIERTE VERSION
train_dataset = create_optimized_tf_dataset(
    x_train_norm, y_train_keras, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    training=True,
    cache_data=True  # 🔹 NEU: Cache aktiviert
)

test_dataset = create_optimized_tf_dataset(
    x_test_norm, y_test_keras, 
    batch_size=BATCH_SIZE, 
    shuffle=False,    # 🔹 WICHTIG: Kein Shuffle für Testdaten!
    training=False,
    cache_data=True   # 🔹 NEU: Cache auch für Testdaten
)

print(f"✅ OPTIMIERTER Trainingsdatensatz erstellt:")
print(f"   - Batch-Größe: {BATCH_SIZE}")
print(f"   - Shuffling: aktiviert") 
print(f"   - Caching: aktiviert 🔹 NEU")
print(f"   - Prefetching: optimiert 🔹 NEU")

print(f"✅ OPTIMIERTER Testdatensatz erstellt:")
print(f"   - Batch-Größe: {BATCH_SIZE}")
print(f"   - Shuffling: deaktiviert (korrekt für Evaluation)")
print(f"   - Caching: aktiviert 🔹 NEU")

# =============================================================================
# 7. ERWEITERTE BATCH-ANALYSE MIT PERFORMANCE-FEATURES
# =============================================================================
print("\n7. 🔄 ERWEITERTE BATCH-ANALYSE")

def inspect_optimized_batches(dataset, num_batches=1):
    """
    Erweiterte Analyse mit Performance-Metriken.
    """
    print("🔹 OPTIMIERTE Batch-Informationen:")
    
    for batch_idx, (batch_images, batch_labels) in enumerate(dataset.take(num_batches)):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  - Images Shape: {batch_images.shape}")
        print(f"  - Labels Shape: {batch_labels.shape}")
        print(f"  - Wertebereich: [{batch_images.numpy().min():.3f}, {batch_images.numpy().max():.3f}]")
        print(f"  - Datentyp: {batch_images.dtype}")
        
        # 🔹 NEU: Performance-relevante Informationen
        print(f"  - Eindeutige Labels im Batch: {len(tf.unique(batch_labels).y)}")
        
        # Zeige die ersten 5 Labels für Stichprobe
        print(f"  - Label-Stichprobe: {batch_labels.numpy()[:5]}")

inspect_optimized_batches(train_dataset)

# =============================================================================
# 8. ZUSAMMENFASSUNG MIT NEUEN FEATURES
# =============================================================================
print("\n" + "=" * 50)
print("🎉 OPTIMIERTE ZUSAMMENFASSUNG")
print("=" * 50)

print(f"✅ MNIST Datensatz erfolgreich mit OPTIMIERTER Pipeline geladen")
print(f"📁 Trainingsdaten: {x_train_norm.shape[0]:,} Bilder")
print(f"📁 Testdaten: {x_test_norm.shape[0]:,} Bilder") 
print(f"📐 Bildgröße: {x_train_norm.shape[1]}x{x_train_norm.shape[2]} Pixel")
print(f"🎯 Anzahl Klassen: 10 (Ziffern 0-9)")

print(f"\n🚀 NEUE OPTIMIERUNGEN:")
print(f"📦 Batch-Größe: {BATCH_SIZE}")
print(f"💾 Caching: aktiviert (Performance+)")
print(f"⚡ Prefetching: AUTOTUNE (Performance+)") 
print(f"🔄 Shuffling: intelligent (Training ja, Test nein)")

print(f"\n💾 Dateien gespeichert:")
print(f"   - mnist_random_samples.png")
print(f"   - mnist_per_class.png")

print(f"\n🎯 Der Datensatz ist jetzt OPTIMAL vorbereitet für Deep Learning!")
print(f"   Entspricht den Best Practices aus den Vorlesungsfolien!")