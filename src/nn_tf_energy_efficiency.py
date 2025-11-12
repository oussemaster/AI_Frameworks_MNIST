import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Energy Efficiency Datensatz laden
# Hinweis: Der Datensatz muss manuell heruntergeladen werden oder ist in UCI verfügbar
# Für dieses Beispiel erstellen wir einen simulierten Datensatz basierend auf der Beschreibung

def load_energy_efficiency_data():
    """
    Lädt den Energy Efficiency Datensatz oder erstellt simulierte Daten.
    In der Praxis würde man hier den echten Datensatz von UCI laden.
    """
    # Simulierte Daten für Demonstrationszwecke
    np.random.seed(42)
    n_samples = 768
    
    # Features basierend auf der Datensatzbeschreibung
    X1 = np.random.uniform(0.5, 7.5, n_samples)  # Relative Compactness
    X2 = np.random.uniform(514, 808, n_samples)  # Surface Area
    X3 = np.random.uniform(245, 416, n_samples)  # Wall Area
    X4 = np.random.uniform(110, 320, n_samples)  # Roof Area
    X5 = np.random.uniform(220, 365, n_samples)  # Overall Height
    X6 = np.random.uniform(2, 5, n_samples)      # Orientation
    X7 = np.random.uniform(0.2, 0.8, n_samples)  # Glazing Area
    X8 = np.random.uniform(1, 5, n_samples)      # Glazing Area Distribution
    
    X = np.column_stack([X1, X2, X3, X4, X5, X6, X7, X8])
    
    # Nichtlineare Zielvariablen (Heating Load und Cooling Load simuliert)
    y_heating = (0.5 * X1 + 0.01 * X2 + 0.02 * X3 + 0.005 * X4 + 
                 0.1 * X5 + 0.05 * X6 + 2 * X7 + 0.2 * X8 +
                 0.1 * X1**2 + 0.001 * np.sin(X2) + 
                 np.random.normal(0, 0.5, n_samples))
    
    y_cooling = (0.3 * X1 + 0.008 * X2 + 0.015 * X3 + 0.004 * X4 + 
                  0.08 * X5 + 0.04 * X6 + 1.5 * X7 + 0.15 * X8 +
                  0.08 * X1**2 + 0.001 * np.cos(X2) + 
                  np.random.normal(0, 0.4, n_samples))
    
    y = np.column_stack([y_heating, y_cooling])
    
    feature_names = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 
                     'Roof_Area', 'Overall_Height', 'Orientation', 
                     'Glazing_Area', 'Glazing_Area_Distribution']
    
    return X, y, feature_names

# Daten laden
X, y, feature_names = load_energy_efficiency_data()

# Auf Heating Load fokussieren (erste Zielvariable)
y = y[:, 0]

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Features normalisieren
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"Trainingsdaten Shape: {X_train_scaled.shape}")
print(f"Testdaten Shape: {X_test_scaled.shape}")

# Nichtlineares Regressionsmodell mit MLP
def create_nonlinear_regression_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        
        layers.Dense(1)  # Output für Regression (keine Activation)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Modell erstellen
model = create_nonlinear_regression_model(X_train_scaled.shape[1])
model.summary()

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Modell trainieren
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Vorhersagen (zurücktransformieren in originale Skala)
y_pred_scaled = model.predict(X_test_scaled).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Metriken berechnen
mse = mean_squared_error(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation auf Testdaten:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Ergebnisse visualisieren
def plot_nonlinear_results(y_true, y_pred, history, feature_names, X_test):
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
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1,0].set_xlabel('Echte Werte (Heating Load)')
    axes[1,0].set_ylabel('Vorhersagen (Heating Load)')
    axes[1,0].set_title(f'Vorhersagen vs. Echte Werte (R² = {r2:.3f})')
    
    # Residual Plot
    residuals = y_true - y_pred
    axes[1,1].scatter(y_pred, residuals, alpha=0.5)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Vorhersagen')
    axes[1,1].set_ylabel('Residuen')
    axes[1,1].set_title('Residual Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Feature Importance durch Permutation
    def permutation_importance(model, X, y, feature_names, n_repeats=10):
        baseline_score = r2_score(y, model.predict(X).flatten())
        importance_scores = []
        
        for i in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                perm_score = r2_score(y, model.predict(X_permuted).flatten())
                scores.append(baseline_score - perm_score)
            
            importance_scores.append(np.mean(scores))
        
        return np.array(importance_scores)
    
    # Feature Importance berechnen und plotten
    importance = permutation_importance(model, X_test_scaled, y_test_scaled, feature_names)
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importance)
    plt.barh(np.array(feature_names)[indices], importance[indices])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance - Nichtlineare Regression')
    plt.tight_layout()
    plt.show()

# Visualisierung
plot_nonlinear_results(y_test, y_pred, history, feature_names, X_test_scaled)