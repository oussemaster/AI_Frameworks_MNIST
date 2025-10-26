# mnist_tf_minimal.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

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