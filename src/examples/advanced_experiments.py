"""
Advanced experiments and examples for quantum neural networks.
This file contains additional functionality that was present in the notebooks.
"""

import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from models.quantum_layers import QuantumLayer, HybridLayer
from utils.quantum_utils import normalize_data, create_bell_state, measure_entanglement

def quantum_feature_mapping():
    """Demonstrate quantum feature mapping with different encodings."""
    # Create quantum device
    n_qubits = 3
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def angle_encoding(x):
        # Angle encoding
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    @qml.qnode(dev)
    def amplitude_encoding(x):
        # Amplitude encoding
        qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Test data
    x = np.random.random(n_qubits)
    x = x / np.linalg.norm(x)  # Normalize for amplitude encoding
    
    print("Quantum Feature Mapping Comparison")
    print(f"Input data: {x}")
    print(f"Angle encoding output: {angle_encoding(x)}")
    print(f"Amplitude encoding output: {amplitude_encoding(x)}")

def quantum_kernel_estimation():
    """Demonstrate quantum kernel estimation."""
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def quantum_kernel(x1, x2):
        # Encode first datapoint
        for i in range(n_qubits):
            qml.RY(x1[i], wires=i)
        
        # Apply inverse encoding of second datapoint
        for i in range(n_qubits):
            qml.RY(-x2[i], wires=i)
        
        # Measure overlap
        return qml.expval(qml.PauliZ(0))
    
    # Generate random data points
    x1 = np.random.random(n_qubits)
    x2 = np.random.random(n_qubits)
    
    # Calculate kernel value
    kernel_value = quantum_kernel(x1, x2)
    print("\nQuantum Kernel Estimation")
    print(f"Data point 1: {x1}")
    print(f"Data point 2: {x2}")
    print(f"Kernel value: {kernel_value}")

def noise_resilience_test():
    """Test quantum neural network resilience to noise."""
    # Create noisy quantum device
    n_qubits = 2
    dev_ideal = qml.device("default.qubit", wires=n_qubits)
    dev_noisy = qml.device("default.mixed", wires=n_qubits)
    
    def create_circuit(dev):
        @qml.qnode(dev)
        def circuit(params, x):
            # Encode input
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
            
            # Variational layer
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return circuit
    
    # Create circuits
    circuit_ideal = create_circuit(dev_ideal)
    circuit_noisy = create_circuit(dev_noisy)
    
    # Test parameters and input
    params = np.random.random(2) * np.pi
    x = np.random.random(2**n_qubits)
    x = x / np.linalg.norm(x)
    
    # Compare results
    result_ideal = circuit_ideal(params, x)
    result_noisy = circuit_noisy(params, x)
    
    print("\nNoise Resilience Test")
    print(f"Ideal circuit output: {result_ideal}")
    print(f"Noisy circuit output: {result_noisy}")
    print(f"Difference: {np.abs(np.array(result_ideal) - np.array(result_noisy))}")

def quantum_gradients_study():
    """Study gradient behavior in quantum neural networks."""
    n_qubits = 2
    quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=2)
    
    # Create test data
    x = torch.randn(1, 2**n_qubits, dtype=torch.float32, requires_grad=True)
    
    # Forward pass
    output = quantum_layer(x)
    loss = torch.sum(output)
    
    # Compute gradients
    loss.backward()
    
    print("\nQuantum Gradients Study")
    print(f"Input gradients:\n{x.grad}")
    print(f"Layer parameter gradients:\n{quantum_layer.weights.grad}")
    
    # Plot gradient distribution
    plt.figure(figsize=(10, 6))
    plt.hist(quantum_layer.weights.grad.numpy().flatten(), bins=30)
    plt.title("Distribution of Quantum Layer Gradients")
    plt.xlabel("Gradient Value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

def main():
    """Run all advanced experiments."""
    print("=== Advanced Quantum Neural Network Experiments ===")
    
    # Run experiments
    quantum_feature_mapping()
    quantum_kernel_estimation()
    noise_resilience_test()
    quantum_gradients_study()

if __name__ == "__main__":
    main() 