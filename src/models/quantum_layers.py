"""
Quantum Neural Network Layer Implementations
This module provides basic quantum layer implementations for neural networks using PennyLane.
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QuantumLayer(nn.Module):
    """
    A quantum layer that can be integrated into PyTorch models.
    This implementation uses PennyLane's quantum nodes with PyTorch interface.
    """
    
    def __init__(self, n_qubits=4, n_layers=2, device="default.qubit"):
        """
        Initialize the quantum layer.
        
        Args:
            n_qubits (int): Number of qubits to use
            n_layers (int): Number of layers in the variational quantum circuit
            device (str): Quantum device to use (default: "default.qubit")
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize the quantum device
        self.dev = qml.device(device, wires=n_qubits)
        
        # Define the variational circuit
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            # Encode classical data into quantum states
            qml.templates.AmplitudeEmbedding(
                inputs, wires=range(self.n_qubits), normalize=True
            )
            
            # Apply parameterized quantum layers
            qml.templates.StronglyEntanglingLayers(
                weights, wires=range(self.n_qubits)
            )
            
            # Return measurement expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # Initialize trainable parameters
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float32, requires_grad=True)
        )
    
    def forward(self, x):
        """
        Forward pass of the quantum layer.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after quantum processing
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flatten input
        
        # Ensure input dimension matches number of qubits
        if x.shape[1] != 2**self.n_qubits:
            raise ValueError(
                f"Input dimension {x.shape[1]} does not match required dimension {2**self.n_qubits}"
            )
        
        # Process each sample in the batch
        results = []
        for sample in x:
            # Ensure input is float32
            sample = sample.to(dtype=torch.float32)
            result = self.quantum_circuit(sample, self.weights)
            results.append(torch.stack(result))
        
        return torch.stack(results)


class HybridLayer(nn.Module):
    """
    A hybrid quantum-classical layer combining both quantum and classical processing.
    """
    
    def __init__(self, in_features, out_features, n_qubits=4, n_layers=2):
        """
        Initialize the hybrid layer.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            n_qubits (int): Number of qubits for the quantum layer
            n_layers (int): Number of layers in the quantum circuit
        """
        super().__init__()
        
        # Initialize all components with float32 dtype
        self.pre_processing = nn.Linear(in_features, 2**n_qubits, dtype=torch.float32)
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.post_processing = nn.Linear(n_qubits, out_features, dtype=torch.float32)
        
        # Initialize weights with float32
        with torch.no_grad():
            self.pre_processing.weight.data = self.pre_processing.weight.data.to(dtype=torch.float32)
            self.pre_processing.bias.data = self.pre_processing.bias.data.to(dtype=torch.float32)
            self.post_processing.weight.data = self.post_processing.weight.data.to(dtype=torch.float32)
            self.post_processing.bias.data = self.post_processing.bias.data.to(dtype=torch.float32)
    
    def forward(self, x):
        """
        Forward pass of the hybrid layer.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after hybrid processing
        """
        # Ensure input is float32
        x = x.to(dtype=torch.float32)
        
        # Pre-processing
        x = self.pre_processing(x)
        x = torch.relu(x)
        
        # Quantum processing
        x = self.quantum_layer(x)
        
        # Post-processing
        x = x.to(dtype=torch.float32)  # Ensure quantum output is float32
        x = self.post_processing(x)
        
        return x 