import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

class MNISTAutoencoder(nn.Module):
    """Autoencoder für MNIST Bilderrekonstruktion"""
    def __init__(self):
        super(MNISTAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 7x7 -> 14x14
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 14x14 -> 28x28
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.Sigmoid()  # Ausgabe zwischen 0 und 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def mask_images(images, mask_ratio=0.6):
    """Maskiert zufällige Pixel in den Bildern"""
    batch_size, channels, height, width = images.shape
    
    # Zufällige Maske erstellen
    mask = torch.rand(batch_size, channels, height, width) > mask_ratio
    masked_images = images * mask
    
    return masked_images, mask, images

def load_mnist_data():
    """Lädt MNIST Daten"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return train_dataset, test_dataset

def train_autoencoder(model, train_loader, val_loader, epochs=20):
    """Trainiert den Autoencoder"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            # Bilder maskieren
            masked_data, _, original_data = mask_images(data, mask_ratio=0.6)
            
            optimizer.zero_grad()
            outputs = model(masked_data)
            loss = criterion(outputs, original_data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, _ in val_loader:
                masked_data, _, original_data = mask_images(data, mask_ratio=0.6)
                outputs = model(masked_data)
                loss = criterion(outputs, original_data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print('-' * 40)
    
    return {'train_losses': train_losses, 'val_losses': val_losses}

def visualize_reconstruction(model, test_loader, num_examples=5):
    """Visualisiert Rekonstruktionen"""
    model.eval()
    
    # Einige Testbeispiele holen
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images[:num_examples]
    
    with torch.no_grad():
        # Bilder maskieren und rekonstruieren
        masked_images, masks, original_images = mask_images(images, mask_ratio=0.6)
        reconstructed = model(masked_images)
    
    # Denormalisieren für Visualisierung
    original_images = (original_images * 0.5) + 0.5
    masked_images = (masked_images * 0.5) + 0.5
    reconstructed = (reconstructed * 0.5) + 0.5
    
    # Plot
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

def main():
    """Hauptfunktion für Autoencoder Training"""
    print("MNIST Autoencoder mit PyTorch")
    print("=" * 40)
    
    # Daten laden
    train_dataset, test_dataset = load_mnist_data()
    
    # DataLoader erstellen
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Validation Split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
    
    # Modell erstellen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Device: {device}")
    
    model = MNISTAutoencoder().to(device)
    print("Autoencoder Architektur:")
    print(model)
    
    # Training
    history = train_autoencoder(model, train_loader, val_loader, epochs=20)
    
    # Rekonstruktionen visualisieren
    visualize_reconstruction(model, test_loader)
    
    # Loss-Verlauf plotten
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Autoencoder Loss Verlauf')
    plt.xlabel('Epoche')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, history

if __name__ == "__main__":
    model, history = main()