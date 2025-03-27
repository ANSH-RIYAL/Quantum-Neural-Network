"""
Interactive demonstration of Quantum Neural Networks.
Run this script to explore and visualize quantum neural network functionality.
"""

import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from models.quantum_layers import QuantumLayer, HybridLayer
from utils.quantum_utils import normalize_data, create_bell_state, measure_entanglement

def plot_quantum_output(output, title):
    """Helper function to plot quantum outputs."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(output)), output.detach().numpy())
    plt.title(title)
    plt.xlabel('Qubit')
    plt.ylabel('Expectation Value')
    plt.grid(True)
    plt.show()

def plot_training_progress(losses):
    """Helper function to plot training progress."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

def demo_quantum_layer():
    """Demonstrate basic quantum layer operations."""
    print("\n=== Quantum Layer Demonstration ===")
    
    # Create a quantum layer
    quantum_layer = QuantumLayer(n_qubits=2, n_layers=2)
    print("Created quantum layer with 2 qubits and 2 layers")
    
    # Create test input
    batch_size = 4
    input_size = 4  # 2^2 for 2 qubits
    x = torch.randn(batch_size, input_size, dtype=torch.float32)
    
    # Get output
    output = quantum_layer(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nSample output:\n{output}")
    
    # Visualize output
    plot_quantum_output(output[0], "Quantum Layer Output (First Sample)")
    input("Press Enter to continue...")

def demo_hybrid_layer():
    """Demonstrate hybrid quantum-classical network."""
    print("\n=== Hybrid Layer Demonstration ===")
    
    # Create hybrid layer
    hybrid_layer = HybridLayer(in_features=3, out_features=2, n_qubits=2)
    print("Created hybrid layer with 3 input features and 2 output features")
    
    # Create test input
    batch_size = 4
    x = torch.randn(batch_size, 3, dtype=torch.float32)
    
    # Get output
    output = hybrid_layer(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nSample output:\n{output}")
    
    # Visualize output
    plot_quantum_output(output[0], "Hybrid Layer Output (First Sample)")
    input("Press Enter to continue...")

def demo_bell_state():
    """Demonstrate Bell state and entanglement."""
    print("\n=== Bell State and Entanglement Demonstration ===")
    
    # Create Bell state
    bell_state = create_bell_state()
    print(f"Created Bell state: {bell_state}")
    
    # Measure entanglement
    entanglement = measure_entanglement(bell_state)
    print(f"\nEntanglement measure: {entanglement:.6f} bits")
    
    # Visualize Bell state
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(bell_state)), np.abs(bell_state)**2)
    plt.title("Bell State Probability Distribution")
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.xticks(range(len(bell_state)), ['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    plt.grid(True)
    plt.show()
    input("Press Enter to continue...")

def demo_training():
    """Demonstrate training of quantum neural network."""
    print("\n=== Training Demonstration ===")
    
    # Create simple dataset
    batch_size = 10
    input_size = 4  # 2^2 for 2 qubits
    X = torch.randn(batch_size, input_size, dtype=torch.float32)
    y = torch.tensor([[1.0, -1.0] if x[0] > 0 else [-1.0, 1.0] for x in X], dtype=torch.float32)
    
    # Create model and optimizer
    model = QuantumLayer(n_qubits=2, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Training loop
    losses = []
    n_epochs = 20
    
    print("Training quantum neural network...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = torch.mean((output - y)**2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    # Plot training progress
    plot_training_progress(losses)
    input("Press Enter to continue...")

def main():
    """Main interactive demonstration."""
    print("Welcome to the Quantum Neural Network Interactive Demo!")
    print("This script will guide you through various quantum computing concepts.")
    
    while True:
        print("\nAvailable demonstrations:")
        print("1. Quantum Layer")
        print("2. Hybrid Layer")
        print("3. Bell State and Entanglement")
        print("4. Training Example")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            demo_quantum_layer()
        elif choice == '2':
            demo_hybrid_layer()
        elif choice == '3':
            demo_bell_state()
        elif choice == '4':
            demo_training()
        elif choice == '5':
            print("\nThank you for exploring quantum neural networks!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main() 