"""
Comparative study of Quantum Neural Networks vs Classical Neural Networks.
Author: Ansh Riyal

This script compares the performance of:
1. QNN vs NN on MNIST binary classification (0 vs 1)
2. QNN vs CNN on Fashion-MNIST (shirts vs t-shirts)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from models.quantum_layers import QuantumLayer, HybridLayer
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassicalNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))

class SimpleCNN(nn.Module):
    def __init__(self, output_size=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, output_size)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        return self.fc(x)

class QuantumNN(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre_processing = nn.Linear(784, 2**n_qubits, dtype=torch.float32)
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.post_processing = nn.Linear(n_qubits, 2, dtype=torch.float32)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.to(dtype=torch.float32)
        x = torch.relu(self.pre_processing(x))
        x = self.quantum_layer(x)
        x = x.to(dtype=torch.float32)
        return self.post_processing(x)

def load_binary_mnist():
    """Load MNIST dataset with only digits 0 and 1."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Filter for digits 0 and 1
    idx = (dataset.targets == 0) | (dataset.targets == 1)
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    return train_dataset, val_dataset

def load_binary_fashion_mnist():
    """Load Fashion-MNIST dataset with only t-shirts and shirts."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    
    # Filter for t-shirts (0) and shirts (6)
    idx = (dataset.targets == 0) | (dataset.targets == 6)
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    dataset.targets[dataset.targets == 6] = 1  # Convert to binary
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    return train_dataset, val_dataset

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, model_name):
    """Train and evaluate a model."""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0
    training_time = 0
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - start_time
        training_time += epoch_time
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{n_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Time: {epoch_time:.2f}s')
    
    # Calculate final metrics
    final_metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions),
        'training_time': training_time,
        'best_val_acc': best_val_acc
    }
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'metrics': final_metrics
    }

def plot_training_curves(results, title):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for model_name, result in results.items():
        ax1.plot(result['train_losses'], label=f'{model_name} (Train)')
        ax1.plot(result['val_losses'], label=f'{model_name} (Val)')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    for model_name, result in results.items():
        ax2.plot(result['train_accuracies'], label=f'{model_name} (Train)')
        ax2.plot(result['val_accuracies'], label=f'{model_name} (Val)')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def print_comparison(results):
    """Print comparison metrics."""
    print("\nModel Comparison:")
    print("-" * 50)
    headers = ["Metric"] + list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'training_time', 'best_val_acc']
    
    # Print header
    print("".join(f"{h:<15}" for h in headers))
    print("-" * (15 * len(headers)))
    
    # Print metrics
    for metric in metrics:
        row = [metric]
        for model_name in results.keys():
            value = results[model_name]['metrics'][metric]
            if metric == 'training_time':
                row.append(f"{value:.2f}s")
            else:
                row.append(f"{value:.4f}")
        print("".join(f"{cell:<15}" for cell in row))

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 32
    n_epochs = 20
    learning_rate = 0.001
    
    # 1. QNN vs NN on MNIST
    print("\nExperiment 1: QNN vs NN on Binary MNIST (0 vs 1)")
    train_dataset, val_dataset = load_binary_mnist()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize models
    classical_nn = ClassicalNN(784).to(device)
    quantum_nn = QuantumNN(n_qubits=6, n_layers=2).to(device)
    
    # Train models
    criterion = nn.CrossEntropyLoss()
    results_mnist = {}
    
    for model, name in [(classical_nn, 'Classical NN'), (quantum_nn, 'Quantum NN')]:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        results_mnist[name] = train_model(
            model, train_loader, val_loader, criterion, optimizer, n_epochs, device, name
        )
    
    plot_training_curves(results_mnist, "MNIST: Classical NN vs Quantum NN")
    print_comparison(results_mnist)
    
    # 2. QNN vs CNN on Fashion-MNIST
    print("\nExperiment 2: QNN vs CNN on Binary Fashion-MNIST (T-shirts vs Shirts)")
    train_dataset, val_dataset = load_binary_fashion_mnist()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize models
    cnn = SimpleCNN().to(device)
    quantum_nn = QuantumNN(n_qubits=6, n_layers=2).to(device)
    
    # Train models
    results_fashion = {}
    
    for model, name in [(cnn, 'CNN'), (quantum_nn, 'Quantum NN')]:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        results_fashion[name] = train_model(
            model, train_loader, val_loader, criterion, optimizer, n_epochs, device, name
        )
    
    plot_training_curves(results_fashion, "Fashion-MNIST: CNN vs Quantum NN")
    print_comparison(results_fashion)

if __name__ == "__main__":
    main() 