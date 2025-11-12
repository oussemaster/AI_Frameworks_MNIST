import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class NonlinearRegressionModel(nn.Module):
    """Nichtlineares Regressionsmodell mit MLP"""
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(NonlinearRegressionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Versteckte Schichten
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Ausgabeschicht
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_energy_efficiency_data():
    """Erstellt Energy Efficiency Daten"""
    np.random.seed(42)
    n_samples = 768
    
    # Features basierend auf Energy Efficiency Datensatz
    X1 = np.random.uniform(0.5, 7.5, n_samples)  # Relative Compactness
    X2 = np.random.uniform(514, 808, n_samples)  # Surface Area
    X3 = np.random.uniform(245, 416, n_samples)  # Wall Area
    X4 = np.random.uniform(110, 320, n_samples)  # Roof Area
    X5 = np.random.uniform(220, 365, n_samples)  # Overall Height
    X6 = np.random.uniform(2, 5, n_samples)      # Orientation
    X7 = np.random.uniform(0.2, 0.8, n_samples)  # Glazing Area
    X8 = np.random.uniform(1, 5, n_samples)      # Glazing Area Distribution
    
    X = np.column_stack([X1, X2, X3, X4, X5, X6, X7, X8])
    
    # Nichtlineare Zielvariable (Heating Load)
    y = (0.5 * X1 + 0.01 * X2 + 0.02 * X3 + 0.005 * X4 + 
         0.1 * X5 + 0.05 * X6 + 2 * X7 + 0.2 * X8 +
         0.1 * X1**2 + 0.001 * np.sin(X2) + 
         np.random.normal(0, 0.5, n_samples))
    
    feature_names = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 
                     'Roof_Area', 'Overall_Height', 'Orientation', 
                     'Glazing_Area', 'Glazing_Area_Distribution']
    
    return X, y, feature_names

def train_nonlinear_model(model, train_loader, val_loader, epochs=100):
    """Trainiert das nichtlineare Modell"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
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
        
        # Learning Rate anpassen
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            print('-' * 40)
    
    return {'train_losses': train_losses, 'val_losses': val_losses}

def plot_nonlinear_results(y_true, y_pred, history, feature_names):
    """Plottet nichtlineare Regressionsergebnisse"""
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
    axes[0,1].set_xlabel('Echte Werte (Heating Load)')
    axes[0,1].set_ylabel('Vorhersagen (Heating Load)')
    axes[0,1].set_title('Vorhersagen vs. Echte Werte')
    axes[0,1].grid(True)
    
    # Residual Plot
    residuals = y_true - y_pred
    axes[1,0].scatter(y_pred, residuals, alpha=0.5)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Vorhersagen')
    axes[1,0].set_ylabel('Residuen')
    axes[1,0].set_title('Residual Plot')
    axes[1,0].grid(True)
    
    # Feature Distribution
    feature_means = np.mean(X, axis=0)
    axes[1,1].barh(feature_names, feature_means)
    axes[1,1].set_xlabel('Durchschnittlicher Wert')
    axes[1,1].set_title('Feature Verteilung')
    
    plt.tight_layout()
    plt.show()

def permutation_importance(model, X, y, feature_names, n_repeats=10):
    """Berechnet Permutation Importance"""
    model.eval()
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        baseline_predictions = model(X_tensor).squeeze().numpy()
        baseline_score = np.mean((y - baseline_predictions) ** 2)
    
    importance_scores = []
    
    for i in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            X_permuted_tensor = torch.FloatTensor(X_permuted)
            with torch.no_grad():
                perm_predictions = model(X_permuted_tensor).squeeze().numpy()
                perm_score = np.mean((y - perm_predictions) ** 2)
            
            scores.append(perm_score - baseline_score)
        
        importance_scores.append(np.mean(scores))
    
    return np.array(importance_scores)

def main():
    """Hauptfunktion für nichtlineare Regression"""
    print("Energy Efficiency Nichtlineare Regression mit PyTorch")
    print("=" * 50)
    
    # Daten laden
    X, y, feature_names = load_energy_efficiency_data()
    
    # Daten aufteilen
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Daten skalieren
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"Trainingsdaten: {X_train_scaled.shape}")
    print(f"Testdaten: {X_test_scaled.shape}")
    
    # PyTorch Tensoren
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
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
    model = NonlinearRegressionModel(input_dim, hidden_dims=[64, 32, 16])
    
    print("\nModellarchitektur:")
    print(model)
    
    # Training
    print("\nStarte Training...")
    history = train_nonlinear_model(model, train_loader, val_loader, epochs=100)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test_tensor).squeeze().numpy()
    
    # Zurücktransformieren
    test_predictions = scaler_y.inverse_transform(
        test_predictions_scaled.reshape(-1, 1)
    ).flatten()
    
    # Metriken berechnen
    mse = np.mean((y_test - test_predictions) ** 2)
    mae = np.mean(np.abs(y_test - test_predictions))
    r2 = 1 - (np.sum((y_test - test_predictions) ** 2) / 
               np.sum((y_test - np.mean(y_test)) ** 2))
    
    print(f"\nEvaluation auf Testdaten:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature Importance
    importance = permutation_importance(model, X_test_scaled, y_test_scaled, feature_names)
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importance)
    plt.barh(np.array(feature_names)[indices], importance[indices])
    plt.xlabel('Permutation Importance (MSE Increase)')
    plt.title('Feature Importance - Nichtlineare Regression')
    plt.tight_layout()
    plt.show()
    
    # Ergebnisse visualisieren
    plot_nonlinear_results(y_test, test_predictions, history, feature_names)
    
    return model, history, mse, mae, r2

if __name__ == "__main__":
    model, history, mse, mae, r2 = main()