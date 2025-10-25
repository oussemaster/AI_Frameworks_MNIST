# mnist_tensorflow.py

# Importe der benötigten Bibliotheken
import tensorflow as tf  # Haupt-Framework für Deep Learning
from tensorflow.keras.datasets import mnist  # MNIST Datensatz
import tensorflow_datasets as tfds  # Alternative Datensatz-Quelle
import numpy as np  # Numerische Berechnungen
import matplotlib.pyplot as plt  # Visualisierung
import random  # Zufallsoperationen
import os  # Betriebssystem-Funktionen

# =============================================================================
# 1. DATENLADEN - Der erste Schritt im ML Workflow
# =============================================================================
print("\n1. 📥 DATENSÄTZE LADEN")

# 🔹 KRITISCHER KI-STEP: Daten acquisition
# Laden des MNIST-Datensatzes aus Keras
# Rückgabe: Tuple von (Trainingsdaten, Testdaten) jeweils mit (Images, Labels)
(x_train_keras, y_train_keras), (x_test_keras, y_test_keras) = mnist.load_data()
print(f"✅ Keras: {x_train_keras.shape[0]:,} Trainingsbilder, {x_test_keras.shape[0]:,} Testbilder")

# 🔹 ALTERNATIVE DATENQUELLE: TensorFlow Datasets
# Parameter: 
# - 'mnist': Name des Datensatzes
# - split: Aufteilung in Train/Test
# - as_supervised: Gibt (image, label) Tuple zurück
# - shuffle_files: Mischt die Dateien beim Laden
ds_train_tfds, ds_test_tfds = tfds.load('mnist', split=['train', 'test'], 
                                       as_supervised=True, shuffle_files=True)
print(f"✅ TFDS: Datensätze erfolgreich geladen")

# =============================================================================
# 2. DATENNORMALISIERUNG - Wichtiger Preprocessing Schritt
# =============================================================================
print("\n2. 🔧 DATENVORBEREITUNG")

def normalize_data(images: np.ndarray) -> np.ndarray:
    """
    Normalisiert Bilddaten auf den Wertebereich [0, 1].
    
    Parameter: images : np.ndarray -> Eingabebilder mit Pixelwerten im Bereich [0, 255]
    Rückgabe: np.ndarray -> Normalisierte Bilder im Bereich [0.0, 1.0] als float32
    """
    return images.astype('float32') / 255.0

# 🔹 WICHTIGER KI-STEP: Daten-Normalisierung
# Warum? Neuronale Netze lernen besser mit normalisierten Daten
# Verhindert numerische Instabilität und beschleunigt Training
x_train_norm = normalize_data(x_train_keras)
x_test_norm = normalize_data(x_test_keras)

print(f"📊 Wertebereich vor Normalisierung: [{x_train_keras.min()}, {x_train_keras.max()}]")
print(f"📊 Wertebereich nach Normalisierung: [{x_train_norm.min():.3f}, {x_train_norm.max():.3f}]")

# =============================================================================
# 3. DATENEXPLORATION - Verstehen der Datenverteilung
# =============================================================================
print("\n3. 📊 DATENSATZ-STATISTIKEN")

def print_dataset_stats(images: np.ndarray, labels: np.ndarray, name: str) -> None:
    """
    Gibt detaillierte Statistiken über den Datensatz aus.
    
    Parameter:
    ----------
    images : np.ndarray -> Array mit Bilddaten
    labels : np.ndarray -> Array mit Labels/Klassen
    name : str -> Bezeichnung des Datensatzes für die Ausgabe
    Rückgabe:  None
    """
    print(f"\n--- {name} ---")
    print(f"Shape: {images.shape}")  # Dimensionen der Daten
    print(f"Datentyp: {images.dtype}")  # Datentyp der Pixelwerte
    print(f"Eindeutige Klassen: {np.unique(labels)}")  # Verfügbare Labels
    
    # 🔹 WICHTIGER KI-STEP: Klassenverteilungsanalyse
    # Stellt sicher, dass alle Klassen gleichmäßig vertreten sind
    unique, counts = np.unique(labels, return_counts=True)
    print("Klassenverteilung:")
    for label, count in zip(unique, counts):
        print(f"  Ziffer {label}: {count:>5} Beispiele ({count/len(labels)*100:.1f}%)")

# Statistiken für beide Datensätze ausgeben
print_dataset_stats(x_train_norm, y_train_keras, "TRAININGSDATEN")
print_dataset_stats(x_test_norm, y_test_keras, "TESTDATEN")

# =============================================================================
# 4. DATENVISUALISIERUNG - Qualitative Datenanalyse
# =============================================================================
print("\n4. 🖼️ VISUALISIERUNG ZUFÄLLIGER BILDER")

def plot_random_samples(images: np.ndarray, labels: np.ndarray, num_samples: int = 10) -> None:
    """
    Erstellt eine Visualisierung von zufällig ausgewählten Beispielbildern.
    
    Parameter:
    images : np.ndarray
        Array mit Bilddaten
    labels : np.ndarray
        Array mit Labels
    num_samples : int, optional
        Anzahl der anzuzeigenden Beispiele (default: 10)
    
    Rückgabe: None
    """
    # 🔹 Zufällige Auswahl für repräsentative Stichprobe
    indices = random.sample(range(len(images)), num_samples)
    
    # Erstelle Subplots mit 2 Reihen und 5 Spalten
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()  # Mache das 2D-Array zu 1D für einfache Iteration
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx], cmap='gray')  # Bild anzeigen
        axes[i].set_title(f'Label: {labels[idx]}', fontsize=12, weight='bold')
        axes[i].axis('off')  # Achsen ausblenden
    
    plt.tight_layout()
    plt.savefig('mnist_random_samples.png', dpi=150, bbox_inches='tight')
    print(f"✅ Zufällige Beispiele gespeichert als 'mnist_random_samples.png'")

# 🔹 WICHTIGER KI-STEP: Qualitative Datenprüfung
# Sicherstellen, dass Daten korrekt geladen und Labels stimmen
plot_random_samples(x_train_norm, y_train_keras)

# =============================================================================
# 5. KLASSENSPEZIFISCHE ANALYSE - Verstehen jeder einzelnen Klasse
# =============================================================================
print("\n5. 🔢 BEISPIEL FÜR JEDE ZIFFER")

def plot_one_per_class(images: np.ndarray, labels: np.ndarray) -> None:
    """
    Zeigt das erste Beispiel für jede Klasse (Ziffer 0-9) mit Pixelwerten.
    
    Parameter:
    images : np.ndarray
        Array mit Bilddaten
    labels : np.ndarray
        Array mit Labels
    
    Rückgabe:  None
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for digit in range(10):
        # 🔹 Finde ersten Index für jede Ziffer
        # np.where gibt Indices zurück wo labels == digit
        idx = np.where(labels == digit)[0][0]
        
        axes[digit].imshow(images[idx], cmap='viridis')
        axes[digit].set_title(f'Ziffer: {digit}', fontsize=14, weight='bold', pad=10)
        axes[digit].axis('off')
        
        # 🔹 DETAILIERTE PIXELANALYSE: Zeige Pixelwerte für besseres Verständnis
        height, width = images[idx].shape
        for i in range(height):
            for j in range(width):
                if images[idx][i, j] > 0.5:  # Nur helle Pixel beschriften für Lesbarkeit
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
# 6. DATENSATZ-VORBEREITUNG - Finale Vorbereitung für Training
# =============================================================================
print("\n6. 🎯 DATENSÄTZE FÜR TRAINING VORBEREITEN")

def create_tf_dataset(images: np.ndarray, labels: np.ndarray, 
                     batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """
    Erstellt einen optimierten TensorFlow Dataset für effizientes Training.
    
    Parameter:
    ----------
    images : np.ndarray -> Array mit Bilddaten
    labels : np.ndarray
        Array mit Labels
    batch_size : int, optional
        Größe der Batches (default: 32)
    shuffle : bool, optional
        Ob die Daten gemischt werden sollen (default: True)
    
    Rückgabe: tf.data.Dataset -> Optimierter TensorFlow Dataset
    """
    # 🔹 WICHTIGER KI-STEP: TensorFlow Dataset erstellen
    # from_tensor_slices: Erstellt Dataset aus numpy Arrays
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        # 🔹 MISCHEN: Verhindert, dass das Modell die Reihenfolge lernt
        dataset = dataset.shuffle(buffer_size=10000)
    
    # 🔹 BATCHING: Teilt Daten in kleine Gruppen für effizientes Training
    dataset = dataset.batch(batch_size)
    
    # 🔹 PREFETCHING: Lädt nächste Batches parallel zum Training
    # AUTOTUNE: TensorFlow wählt optimale Anzahl automatisch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Parameter für das Training
BATCH_SIZE = 64  # Anzahl Bilder pro Trainingsschritt

# 🔹 WICHTIGER KI-STEP: Finale Datensatzerstellung
# Trainingsdatensatz: Gemischt für bessere Generalisierung
train_dataset = create_tf_dataset(x_train_norm, y_train_keras, 
                                 batch_size=BATCH_SIZE, shuffle=True)
# Testdatensatz: Nicht mischen für konsistente Evaluation
test_dataset = create_tf_dataset(x_test_norm, y_test_keras, 
                                batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Trainingsdatensatz: Batch-Größe {BATCH_SIZE}, shuffling aktiviert")
print(f"✅ Testdatensatz: Batch-Größe {BATCH_SIZE}, shuffling deaktiviert")

# =============================================================================
# 7. BATCH-VERIFIZIERUNG - Sicherstellen der korrekten Verarbeitung
# =============================================================================
print("\n7. 🔄 BATCH-VERARBEITUNG TESTEN")

def inspect_batches(dataset: tf.data.Dataset, num_batches: int = 1) -> None:
    """
    Untersucht die Struktur und Inhalte der Batches im Dataset.
    
    Parameter:
    ----------
    dataset : tf.data.Dataset
        TensorFlow Dataset zur Untersuchung
    num_batches : int, optional
        Anzahl der zu untersuchenden Batches (default: 1)
    
    Rückgabe:
    --------
    None
    """
    print("Erste Batch-Informationen:")
    
    # 🔹 DATASET-ITERATION: take() nimmt die ersten n Batches
    for batch_idx, (batch_images, batch_labels) in enumerate(dataset.take(num_batches)):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  - Images Shape: {batch_images.shape}")  # [batch_size, 28, 28]
        print(f"  - Labels Shape: {batch_labels.shape}")  # [batch_size]
        print(f"  - Labels im Batch: {batch_labels.numpy()}")  # Tatsächliche Labels
        print(f"  - Wertebereich: [{batch_images.numpy().min():.3f}, {batch_images.numpy().max():.3f}]")

# 🔹 QUALITÄTSKONTROLLE: Verifizieren der Batch-Verarbeitung
inspect_batches(train_dataset)

# =============================================================================
# 8. ZUSAMMENFASSUNG - Finaler Statusbericht
# =============================================================================
print("\n" + "=" * 50)
print("🎉 ZUSAMMENFASSUNG")
print("=" * 50)

print(f"✅ MNIST Datensatz erfolgreich geladen und vorbereitet")
print(f"📁 Trainingsdaten: {x_train_norm.shape[0]:,} Bilder")
print(f"📁 Testdaten: {x_test_norm.shape[0]:,} Bilder") 
print(f"📐 Bildgröße: {x_train_norm.shape[1]}x{x_train_norm.shape[2]} Pixel")
print(f"🎯 Anzahl Klassen: 10 (Ziffern 0-9)")
print(f"📦 Batch-Größe: {BATCH_SIZE}")
print(f"💾 Dateien gespeichert: mnist_random_samples.png, mnist_per_class.png")

print("\n🚀 Der Datensatz ist jetzt bereit für Deep Learning Modelle!")