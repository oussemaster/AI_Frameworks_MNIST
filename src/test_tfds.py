# test_tfds.py
import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow Version:", tf.__version__)
print("TensorFlow Datasets Version:", tfds.__version__)

# Teste das Laden eines kleinen Datensatzes
print("Verfügbare Datensätze:")
print(tfds.list_builders()[:5])  # Zeige die ersten 5 verfügbaren Datensätze

print("✅ TensorFlow Datasets funktioniert!")   