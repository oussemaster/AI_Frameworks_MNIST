import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# California Housing Datensatz laden
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Features normalisieren (wichtig für lineare Modelle)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Trainingsdaten Shape: {X_train_scaled.shape}")
print(f"Testdaten Shape: {X_test_scaled.shape}")
print(f"Anzahl Features: {X_train_scaled.shape[1]}")

# Lineares Regressionsmodell mit Sequential API
def create_linear_regression_model(input_dim):
    model = keras.Sequential([
        layers.Dense(1, input_shape=(input_dim,), name='linear_layer')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Modell erstellen
model = create_linear_regression_model(X_train_scaled.shape[1])
model.summary()

# Callback für frühes Stoppen
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

# Modell trainieren
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluation
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Vorhersagen
y_pred = model.predict(X_test_scaled).flatten()

# Ergebnisse visualisieren
def plot_regression_results(y_true, y_pred, history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss-Verlauf
    axes[0,0].plot(history.history['loss'], label='Training Loss')
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,0].set_title('MSE Loss Verlauf')
    axes[0,0].set_xlabel('Epoche')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].legend()
    
    # MAE-Verlauf
    axes[0,1].plot(history.history['mae'], label='Training MAE')
    axes[0,1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0,1].set_title('MAE Verlauf')
    axes[0,1].set_xlabel('Epoche')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].legend()
    
    # Vorhersagen vs. echte Werte
    axes[1,0].scatter(y_true, y_pred, alpha=0.5)
    axes[1,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1,0].set_xlabel('Echte Werte')
    axes[1,0].set_ylabel('Vorhersagen')
    axes[1,0].set_title('Vorhersagen vs. Echte Werte')
    
    # Residual Plot
    residuals = y_true - y_pred
    axes[1,1].scatter(y_pred, residuals, alpha=0.5)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Vorhersagen')
    axes[1,1].set_ylabel('Residuen')
    axes[1,1].set_title('Residual Plot')
    
    plt.tight_layout()
    plt.show()

# Visualisierung
plot_regression_results(y_test, y_pred, history)

# Feature Importance (Gewichte des linearen Modells)
weights = model.get_weights()[0].flatten()
feature_names = housing.feature_names

plt.figure(figsize=(10, 6))
plt.barh(feature_names, weights)
plt.xlabel('Gewicht')
plt.title('Feature Importance - Lineare Regression')
plt.tight_layout()
plt.show()