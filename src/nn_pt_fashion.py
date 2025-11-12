import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

class FashionMNISTCNN(nn.Module):
    """CNN für Fashion-MNIST Klassifikation"""
    def __init__(self, num_classes=10):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)  # 28x28 -> nach 2x Pooling 7x7
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_fashion_mnist_data():
    """Lädt und vorverarbeitet Fashion-MNIST Daten"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Trainingsdaten
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Testdaten
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return train_dataset, test_dataset

def train_model(model, train_loader, val_loader, epochs=10):
    """Trainiert das Modell"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        # Metriken speichern
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, test_loader):
    """Evaluiert das Modell auf Testdaten"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Genauigkeit: {accuracy:.2f}%')
    return accuracy

def plot_results(history, model, test_loader, class_names):
    """Plottet Trainingsverlauf und Beispielvorhersagen"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trainingsverlauf
    axes[0,0].plot(history['train_losses'], label='Training Loss')
    axes[0,0].plot(history['val_losses'], label='Validation Loss')
    axes[0,0].set_title('Modell Loss')
    axes[0,0].set_xlabel('Epoche')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    
    axes[0,1].plot(history['train_accuracies'], label='Training Accuracy')
    axes[0,1].plot(history['val_accuracies'], label='Validation Accuracy')
    axes[0,1].set_title('Modell Genauigkeit')
    axes[0,1].set_xlabel('Epoche')
    axes[0,1].set_ylabel('Genauigkeit (%)')
    axes[0,1].legend()
    
    # Beispielvorhersagen
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    for i in range(4):
        ax = axes[1, i//2] if i < 2 else axes[1, i-2]
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        
        predicted_class = predictions[i].item()
        true_class = labels[i].item()
        
        color = 'green' if predicted_class == true_class else 'red'
        ax.set_title(f'Pred: {class_names[predicted_class]}\nTrue: {class_names[true_class]}', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Hauptfunktion für Fashion-MNIST Klassifikation"""
    print("Fashion-MNIST CNN Klassifikation mit PyTorch")
    print("=" * 50)
    
    # Klassenbezeichnungen
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Daten laden
    train_dataset, test_dataset = load_fashion_mnist_data()
    
    # DataLoader erstellen
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Validation Split (80% Training, 20% Validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    # Modell erstellen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Device: {device}")
    
    model = FashionMNISTCNN().to(device)
    print("Modellarchitektur:")
    print(model)
    
    # Training
    history = train_model(model, train_loader, val_loader, epochs=10)
    
    # Evaluation auf Testdaten
    test_accuracy = evaluate_model(model, test_loader)
    
    # Ergebnisse visualisieren
    plot_results(history, model, test_loader, class_names)
    
    return model, history, test_accuracy

if __name__ == "__main__":
    model, history, test_accuracy = main()