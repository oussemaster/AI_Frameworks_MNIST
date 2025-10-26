# mnist_tf_minimal.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# 1. Von Keras laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalisierung
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. tf.data.Dataset erstellen
def create_tf_dataset(images, labels, batch_size=32, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Paralleles Laden
    return dataset

# 4. Datensätze erstellen
train_ds = create_tf_dataset(x_train, y_train, training=True)
test_ds = create_tf_dataset(x_test, y_test, training=False)

print("✅ TensorFlow MNIST ready!")
print(f"Training batches: {len(list(train_ds))}")

# 5. Beispielbilder anzeigen
def show_sample_images(images, labels, num_images=5):
    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('tf_mnist_samples.png', dpi=100, bbox_inches='tight')
    plt.show()

# Ersten Batch aus Dataset extrahieren und anzeigen
for batch_images, batch_labels in train_ds.take(1):
    # Konvertiere Tensor zu NumPy für Visualisierung
    images_np = batch_images.numpy()
    labels_np = batch_labels.numpy()
    
    print("✅ TensorFlow MNIST ready!")
    print(f"Batch Shape: {images_np.shape}")
    print(f"Labels: {labels_np[:5]}")
    
    # Zeige erste 5 Bilder des Batches
    show_sample_images(images_np, labels_np)

print(f"Training batches: {len(list(train_ds))}")
print("💾 Sample images saved as 'tf_mnist_samples.png'")

