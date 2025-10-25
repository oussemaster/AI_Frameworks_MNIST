# test_tf.py - Minimal Test
import tensorflow as tf
import numpy as np

print("TensorFlow Version:", tf.__version__)
print("NumPy Version:", np.__version__)

# Einfacher Test ohne MNIST
print("Erstelle einfachen Tensor...")
x = tf.constant([[1, 2], [3, 4]])
print("Tensor:", x)
print("Shape:", x.shape)

print("✅ TensorFlow Grundfunktionalität funktioniert!")