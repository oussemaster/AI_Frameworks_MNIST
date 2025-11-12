import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# MNIST Datensatz laden
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Daten normalisieren und dimensionieren
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Trainingsdaten Shape: {x_train.shape}")

# Funktion zum Maskieren von Pixeln
def mask_images(images, mask_ratio=0.5):
    """Maskiert zufällige Pixel in den Bildern"""
    batch_size = images.shape[0]
    height, width, channels = images.shape[1:]
    
    # Zufällige Maske erstellen
    mask = np.random.random((batch_size, height, width, channels)) > mask_ratio
    masked_images = images * mask
    
    return masked_images, mask, images

# Autoencoder mit Functional API definieren
def create_autoencoder():
    # Encoder
    inputs = keras.Input(shape=(28, 28, 1))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck
    encoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Modell erstellen
    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder

# Autoencoder erstellen und anzeigen
autoencoder = create_autoencoder()
autoencoder.summary()

# Training vorbereiten
masked_x_train, _, original_x_train = mask_images(x_train, mask_ratio=0.6)
masked_x_test, _, original_x_test = mask_images(x_test, mask_ratio=0.6)

# Autoencoder trainieren
history = autoencoder.fit(
    masked_x_train, original_x_train,
    epochs=20,
    batch_size=128,
    shuffle=True,
    validation_data=(masked_x_test, original_x_test)
)

# Rekonstruktion testen und visualisieren
def visualize_reconstruction(autoencoder, x_test, num_examples=5):
    # Bilder maskieren
    masked_images, masks, original_images = mask_images(x_test[:num_examples])
    
    # Rekonstruktion
    reconstructed = autoencoder.predict(masked_images)
    
    # Visualisierung
    fig, axes = plt.subplots(num_examples, 4, figsize=(12, 3*num_examples))
    
    for i in range(num_examples):
        # Original
        axes[i, 0].imshow(original_images[i].squeeze(), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Maske
        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Maske')
        axes[i, 1].axis('off')
        
        # Maskiertes Bild
        axes[i, 2].imshow(masked_images[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Maskiert')
        axes[i, 2].axis('off')
        
        # Rekonstruktion
        axes[i, 3].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[i, 3].set_title('Rekonstruiert')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

# Ergebnisse visualisieren
visualize_reconstruction(autoencoder, x_test)

# Loss-Verlauf plotten
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss')
plt.xlabel('Epoche')
plt.ylabel('MSE Loss')
plt.legend()

plt.tight_layout()
plt.show()