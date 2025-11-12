import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Fashion-MNIST Datensatz laden
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Klassenbezeichnungen für Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Daten vorverarbeiten
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Dimension für CNN erweitern (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Trainingsdaten Shape: {x_train.shape}")
print(f"Testdaten Shape: {x_test.shape}")

# CNN Modell mit Sequential API definieren
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    
    # Erste Faltungsblock
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Zweite Faltungsblock
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Dritte Faltungsblock
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    
    # Klassifikator
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Regularisierung
    layers.Dense(10, activation='softmax')
])

# Modell kompilieren
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Modellarchitektur anzeigen
model.summary()

# Training mit Validation Split
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# Evaluation auf Testdaten
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Genauigkeit: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Visualisierung der Ergebnisse
def plot_results(history, model, x_test, y_test, class_names):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trainingsverlauf
    axes[0,0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0,0].set_title('Modell Genauigkeit')
    axes[0,0].set_xlabel('Epoche')
    axes[0,0].set_ylabel('Genauigkeit')
    axes[0,0].legend()
    
    axes[0,1].plot(history.history['loss'], label='Training Loss')
    axes[0,1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,1].set_title('Modell Loss')
    axes[0,1].set_xlabel('Epoche')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    
    # Vorhersagen auf Testdaten
    predictions = model.predict(x_test)
    
    # Einige Beispiele visualisieren
    for i in range(4):
        idx = np.random.randint(0, len(x_test))
        axes[1,0 if i<2 else 1].imshow(x_test[idx].squeeze(), cmap='gray')
        predicted_class = np.argmax(predictions[idx])
        true_class = y_test[idx]
        
        color = 'green' if predicted_class == true_class else 'red'
        axes[1,0 if i<2 else 1].set_title(
            f'Pred: {class_names[predicted_class]}\nTrue: {class_names[true_class]}',
            color=color
        )
        axes[1,0 if i<2 else 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Ergebnisse plotten
plot_results(history, model, x_test, y_test, class_names)