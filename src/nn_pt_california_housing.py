import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from urllib.request import urlretrieve
import os

class LinearRegressionModel(nn.Module):
    """Einfaches lineares Regressionsmodell"""
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def load_california_housing():
    """Lädt California Housing Daten"""
    try:
        california_housing_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        
        if not os.path.exists('california_housing.csv'):
            print("Lade California Housing Datensatz herunter...")
            urlretrieve(california_housing_url, 'california_housing.csv')
        
        housing = pd.read_csv('california_housing.csv')
        housing = housing.dropna()
        
        feature_columns = ['longitude', 'latitude', 'housing_median_age', 
                          'total_rooms', 'total_bedrooms', 'population', 
                          'households', 'median_income']
        
        X = housing[feature_columns].values
        y = housing['median_house_value'].values
        
        print(f"Echter Datensatz geladen: {X.shape[0]} Samples")
        return X, y, feature_columns
        
    except Exception as e:
        print(f"Download fehlgeschlagen: {e}")
        return load_simulated_housing_data()

def load_simulated_housing_data():
    """Erstellt simulierte Housing Daten"""
    np.random.seed(42)
    n_samples = 20000
    
    longitude = np.random.uniform(-124.3, -114.3, n_samples)
    latitude = np.random.uniform(32.5, 42, n_samples)
    housing_median_age = np.random.uniform(1, 52, n_samples)
    total_rooms = np.random.uniform(2, 40000, n_samples)
    total_bedrooms = np.random.uniform(1, 6500, n_samples)
    population = np.random.uniform(3, 36000, n_samples)
    households = np.random.uniform(1, 6000, n_samples)
    median_income = np.random.uniform(0.5, 15, n_samples)
    
    X = np.column_stack([
        longitude, latitude, housing_median_age, total_rooms,
        total_bedrooms, population, households, median_income
    ])
    
    y = (
        100000 + 
        5000 * median_income + 
        1000 * housing_median_age -
        10 * population +
        50 * total_rooms +
        np.random.normal(0, 50000, n_samples)
    )
    
    feature_columns = ['longitude', 'latitude', 'housing_median_age', 
                      'total_rooms', 'total_bedrooms', 'population', 
                      'households', 'median_income']
    
    print(f"Simulierte Daten erstellt: {X.shape[0]} Samples")
    return X, y, feature_columns

def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    """Manueller Train-Test Split"""
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def custom_standard_scaler(X_train, X_test):
    """Manuelle StandardScaler Implementierung"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1, std)
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled, mean, std

def train_regression_model(model, train_loader, val_loader, epochs=100):
    """Trainiert das Regressionsmodell"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.2f}')
            print(f'  Val Loss: {val_loss:.2f}')
    
    return {'train_losses': train_losses, 'val_losses': val_losses}

def plot_regression_results(y_true, y_pred, history, feature_names, model):
    """Plottet Regressionsergebnisse"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss-Verlauf
    axes[0,0].plot(history['train_losses'], label='Training Loss')
    axes[0,0].plot(history['val_losses'], label='Validation Loss')
    axes[0,0].set_title('MSE Loss Verlauf')
    axes[0,0].set_xlabel('Epoche')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Vorhersagen vs. echte Werte
    axes[0,1].scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0,1].set_xlabel('Echte Werte ($)')
    axes[0,1].set_ylabel('Vorhersagen ($)')
    axes[0,1].set_title('Vorhersagen vs. Echte Werte')
    axes[0,1].grid(True)
    
    # Residual Plot
    residuals = y_true - y_pred
    axes[1,0].scatter(y_pred, residuals, alpha=0.5)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Vorhersagen ($)')
    axes[1,0].set_ylabel('Residuen ($)')
    axes[1,0].set_title('Residual Plot')
    axes[1,0].grid(True)
    
    # Feature Importance
    weights = model.linear.weight.detach().numpy().flatten()
    axes[1,1].barh(feature_names, weights)
    axes[1,1].set_xlabel('Gewicht')
    axes[1,1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.show()

def main():
    """Hauptfunktion für lineare Regression"""
    print("California Housing Lineare Regression mit PyTorch")
    print("=" * 50)
    
    # Daten laden
    X, y, feature_names = load_california_housing()
    
    # Daten aufteilen und skalieren
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, mean, std = custom_standard_scaler(X_train, X_test)
    
    print(f"Trainingsdaten: {X_train_scaled.shape}")
    print(f"Testdaten: {X_test_scaled.shape}")
    
    # PyTorch Tensoren erstellen
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Datensätze und DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Validation Split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Modell erstellen
    input_dim = X_train_scaled.shape[1]
    model = LinearRegressionModel(input_dim)
    
    print("\nModellarchitektur:")
    print(model)
    
    # Training
    print("\nStarte Training...")
    history = train_regression_model(model, train_loader, val_loader, epochs=100)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze().numpy()
    
    # Metriken berechnen
    mse = np.mean((y_test - test_predictions) ** 2)
    mae = np.mean(np.abs(y_test - test_predictions))
    
    print(f"\nEvaluation auf Testdaten:")
    print(f"MSE: {mse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    
    # Ergebnisse visualisieren
    plot_regression_results(y_test, test_predictions, history, feature_names, model)
    
    return model, history, mse, mae

if __name__ == "__main__":
    model, history, mse, mae = main()