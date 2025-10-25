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
    ----------
    images : np.ndarray
        Array mit Bilddaten
    labels : np.ndarray
        Array mit Labels
    num_samples : int, optional
        Anzahl der anzuzeigenden Beispiele (default: 10)
    
    Rückgabe:
    --------
    None
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
    ----------
    images : np.ndarray
        Array mit Bilddaten
    labels : np.ndarray
        Array mit Labels
    
    Rückgabe:
    --------
    None
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
    images : np.ndarray
        Array mit Bilddaten
    labels : np.ndarray
        Array mit Labels
    batch_size : int, optional
        Größe der Batches (default: 32)
    shuffle : bool, optional
        Ob die Daten gemischt werden sollen (default: True)
    
    Rückgabe:
    --------
    tf.data.Dataset
        Optimierter TensorFlow Dataset
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

# =============================================================================
# 9. KI-MODELL ERSTELLEN UND TRAINIEREN
# =============================================================================
print("\n9. 🧠 KI-MODELL TRAINIEREN")

def create_and_train_model(x_train, y_train, x_test, y_test):
    """
    Erstellt und trainiert ein CNN-Modell für die Ziffernerkennung.
    
    Parameter:
    ----------
    x_train : np.ndarray - Trainingsbilder
    y_train : np.ndarray - Trainingslabels  
    x_test : np.ndarray - Testbilder
    y_test : np.ndarray - Testlabels
    
    Rückgabe:
    --------
    tf.keras.Model - Trainiertes Modell
    dict - Trainingsverlauf
    """
    
    # 🔹 WICHTIG: Daten für CNN vorbereiten (Channel Dimension hinzufügen)
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)  # Shape: (60000, 28, 28, 1)
    x_test_cnn = x_test.reshape(-1, 28, 28, 1)    # Shape: (10000, 28, 28, 1)
    
    # 🔹 Labels in kategorisches Format umwandeln (one-hot encoding)
    y_train_categorical = tf.keras.utils.to_categorical(y_train, 10)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)
    
    print("📐 Modell-Input Shape:", x_train_cnn.shape)
    print("🎯 Modell-Output Shape:", y_train_categorical.shape)
    
    # 🔹 CONVOLUTIONAL NEURAL NETWORK (CNN) erstellen
    model = tf.keras.Sequential([
        # Erste Convolutional Layer - lernt grundlegende Muster (Kanten, Ecken)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Zweite Convolutional Layer - lernt komplexere Muster (Kreise, Kurven)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Dritte Convolutional Layer - lernt hochlevel Merkmale
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # 🔹 Vektorisierung für Fully-Connected Layers
        tf.keras.layers.Flatten(),
        
        # 🔹 Fully-Connected Layers für Klassifikation
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Verhindert Overfitting
        
        # 🔹 Output Layer - 10 Neuronen für 10 Ziffern (0-9)
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 🔹 MODELL KOMPILIEREN
    model.compile(
        optimizer='adam',           # Optimierungsalgorithmus
        loss='categorical_crossentropy',  # Verlustfunktion für Klassifikation
        metrics=['accuracy']        # Metrik zur Leistungsbewertung
    )
    
    print("✅ Modell erfolgreich kompiliert")
    print(model.summary())
    
    # 🔹 MODELL TRAINIEREN
    print("🚀 Starte Training...")
    history = model.fit(
        x_train_cnn, y_train_categorical,
        epochs=10,                  # Anzahl der Trainingsdurchläufe
        batch_size=32,              # Bilder pro Update
        validation_split=0.2,       # 20% für Validation
        verbose=1
    )
    
    # 🔹 MODELL EVALUIEREN
    test_loss, test_accuracy = model.evaluate(x_test_cnn, y_test_categorical, verbose=0)
    print(f"🎯 Test-Genauigkeit: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return model, history

# Modell erstellen und trainieren
trained_model, training_history = create_and_train_model(
    x_train_norm, y_train_keras, 
    x_test_norm, y_test_keras
)

# Modell speichern für spätere Verwendung
trained_model.save('mnist_cnn_model.h5')
print("💾 Modell gespeichert als 'mnist_cnn_model.h5'")

# =============================================================================
# 10. VORHERSAGE FÜR NEUE BILDER
# =============================================================================
print("\n10. 🔮 VORHERSAGE FÜR EIGENE BILDER")

def predict_custom_image(model, image_path):
    """
    Macht eine Vorhersage für ein eigenes Zahlenbild.
    
    Parameter:
    ----------
    model : tf.keras.Model - Trainiertes Modell
    image_path : str - Pfad zum Bild
    
    Rückgabe:
    --------
    int - Vorhergesagte Ziffer (0-9)
    np.ndarray - Wahrscheinlichkeiten für alle Ziffern
    """
    
    try:
        # 🔹 Bild laden und vorbereiten
        from PIL import Image
        import numpy as np
        
        # Bild in Graustufen laden und auf 28x28 skalieren
        img = Image.open(image_path).convert('L')  # Zu Graustufen
        img = img.resize((28, 28))                 # Auf MNIST-Größe skalieren
        
        # Zu Numpy-Array konvertieren
        img_array = np.array(img)
        
        # 🔹 WICHTIG: Invertieren falls nötig (MNIST ist weiß auf schwarz)
        # Wenn dein Bild schwarze Ziffern auf weißem Hintergrund hat:
        if np.mean(img_array) > 127:  # Heller Hintergrund
            img_array = 255 - img_array  # Invertieren
        
        # Normalisieren und für Modell vorbereiten
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  # Batch-Dimension hinzufügen
        
        # 🔹 VORHERSAGE MACHT
        predictions = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        print(f"🔍 Vorhersage für {image_path}:")
        print(f"   🎯 Ziffer: {predicted_digit}")
        print(f"   📊 Sicherheit: {confidence:.2%}")
        
        # Zeige Top-3 Vorhersagen
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        print("   📈 Top-3 Vorhersagen:")
        for i, idx in enumerate(top_3_indices):
            print(f"      {i+1}. Ziffer {idx}: {predictions[0][idx]:.2%}")
        
        return predicted_digit, predictions[0]
        
    except Exception as e:
        print(f"❌ Fehler beim Verarbeiten des Bildes: {e}")
        return None, None

# Beispiel für die Verwendung:
# predicted, probs = predict_custom_image(trained_model, 'meine_zahl.png')

# =============================================================================
# 11. EINFACHE BENUTZEROBERFLÄCHE
# =============================================================================
print("\n11. 🖱️ BENUTZEROBERFLÄCHE")

def interactive_prediction():
    """
    Bietet eine interaktive Möglichkeit, eigene Bilder zu klassifizieren.
    """
    print("\n" + "="*50)
    print("INTERAKTIVE ZIFFERNERKENNUNG")
    print("="*50)
    print("So verwendest du das Modell:")
    print("1. Erstelle ein Bild einer handgeschriebenen Ziffer (0-9)")
    print("2. Speichere es als PNG/JPG (empfohlen: 28x28 Pixel)")
    print("3. Gib den Pfad zum Bild ein")
    print("4. Das Modell sagt die Ziffer vorher!")
    print("\nTipp: Verwende Paint, Photoshop oder zeichne auf deinem Handy")
    
    while True:
        print("\n--- Neue Vorhersage ---")
        image_path = input("Pfad zum Bild (oder 'quit' zum Beenden): ").strip()
        
        if image_path.lower() == 'quit':
            break
            
        if not os.path.exists(image_path):
            print("❌ Datei existiert nicht! Bitte Pfad überprüfen.")
            continue
            
        # Vorhersage machen
        predicted, probabilities = predict_custom_image(trained_model, image_path)
        
        if predicted is not None:
            # Bild anzeigen
            img = Image.open(image_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f'Vorhersage: Ziffer {predicted}', fontsize=16, weight='bold')
            plt.axis('off')
            plt.show()

# 🔹 BENUTZEROBERFLÄCHE STARTEN (auskommentiert, da sie Input benötigt)
# interactive_prediction()

