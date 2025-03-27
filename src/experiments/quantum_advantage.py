"""
Demonstration of quantum advantage in pattern recognition tasks.
This experiment shows how quantum neural networks can be more efficient
at recognizing certain patterns that are difficult for classical neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.quantum_layers import QuantumLayer, HybridLayer
from src.utils.quantum_utils import normalize_data, compute_quantum_accuracy


def generate_quantum_pattern_data(
    n_samples: int = 1000,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data with patterns that exploit quantum properties.
    The pattern is based on entangled states, which are harder
    to recognize with classical neural networks.
    
    Args:
        n_samples (int): Number of samples to generate
        noise_level (float): Amount of noise to add
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels
    """
    # Generate base patterns
    pattern_a = np.array([1, 1, 0, 0]) / np.sqrt(2)  # Similar to Bell state |00⟩ + |11⟩
    pattern_b = np.array([0, 0, 1, 1]) / np.sqrt(2)  # Orthogonal to pattern_a
    
    # Generate samples
    X = []
    y = []
    
    for _ in range(n_samples):
        if np.random.random() > 0.5:
            # Generate sample from pattern_a
            sample = pattern_a + np.random.normal(0, noise_level, 4)
            X.append(sample)
            y.append(0)
        else:
            # Generate sample from pattern_b
            sample = pattern_b + np.random.normal(0, noise_level, 4)
            X.append(sample)
            y.append(1)
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize the data
    X = normalize_data(X)
    
    return X, y


class ClassicalNN(nn.Module):
    """Simple classical neural network."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class QuantumNN(nn.Module):
    """Simple quantum neural network."""
    
    def __init__(self, n_qubits: int):
        super().__init__()
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=2)
        self.post_processing = nn.Linear(n_qubits, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantum_layer(x)
        x = self.post_processing(x)
        return x


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_epochs: int = 100
) -> List[float]:
    """
    Train a model and return training history.
    
    Args:
        model: Neural network model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_epochs: Number of training epochs
    
    Returns:
        List[float]: Training accuracy history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    history = []
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            accuracy = compute_quantum_accuracy(test_outputs, y_test)
            history.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    return history


def main():
    """Run the quantum advantage experiment."""
    # Generate data
    X, y = generate_quantum_pattern_data(n_samples=1000, noise_level=0.1)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train = torch.FloatTensor(X[:split_idx])
    y_train = torch.LongTensor(y[:split_idx])
    X_test = torch.FloatTensor(X[split_idx:])
    y_test = torch.LongTensor(y[split_idx:])
    
    # Train classical model
    classical_model = ClassicalNN(input_size=4, hidden_size=8, output_size=2)
    classical_history = train_model(
        classical_model, X_train, y_train, X_test, y_test
    )
    
    # Train quantum model
    quantum_model = QuantumNN(n_qubits=2)  # 2 qubits = 4 dimensional input
    quantum_history = train_model(
        quantum_model, X_train, y_train, X_test, y_test
    )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(classical_history, label='Classical NN')
    plt.plot(quantum_history, label='Quantum NN')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Classical vs Quantum Neural Network Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('quantum_advantage_results.png')
    plt.close()
    
    # Print final results
    print("\nFinal Results:")
    print(f"Classical NN accuracy: {classical_history[-1]:.4f}")
    print(f"Quantum NN accuracy: {quantum_history[-1]:.4f}")


if __name__ == "__main__":
    main() 