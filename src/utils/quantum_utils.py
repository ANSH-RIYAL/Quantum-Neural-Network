"""
Utility functions for quantum neural networks.
"""

import numpy as np
import torch
import pennylane as qml
from typing import List, Tuple, Union


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to be suitable for quantum state preparation.
    
    Args:
        data (np.ndarray): Input data
    
    Returns:
        np.ndarray: Normalized data
    """
    return data / np.sqrt(np.sum(data**2, axis=1, keepdims=True))


def binary_to_angles(binary_string: str) -> List[float]:
    """
    Convert a binary string to rotation angles for quantum state preparation.
    
    Args:
        binary_string (str): Binary string to convert
    
    Returns:
        List[float]: List of rotation angles
    """
    n = len(binary_string)
    angles = []
    
    for i in range(n):
        if binary_string[i] == '1':
            angles.append(np.pi)
        else:
            angles.append(0)
    
    return angles


def compute_quantum_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute accuracy for quantum model predictions.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): True labels
    
    Returns:
        float: Accuracy score
    """
    with torch.no_grad():
        predicted_labels = torch.argmax(predictions, dim=1)
        correct = (predicted_labels == targets).sum().item()
        total = targets.size(0)
        return correct / total


def create_bell_state() -> np.ndarray:
    """
    Create a Bell state (maximally entangled state of two qubits).
    
    Returns:
        np.ndarray: Quantum state vector
    """
    dev = qml.device('default.qubit', wires=2)
    
    @qml.qnode(dev)
    def bell_state():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()
    
    return bell_state()


def quantum_fisher_information(
    circuit: callable,
    params: np.ndarray,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Compute the Quantum Fisher Information matrix for a parametrized quantum circuit.
    
    Args:
        circuit (callable): Quantum circuit function
        params (np.ndarray): Circuit parameters
        eps (float): Small parameter for finite difference
    
    Returns:
        np.ndarray: Quantum Fisher Information matrix
    """
    n_params = len(params)
    qfim = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            # Compute partial derivatives
            params_plus_i = params.copy()
            params_plus_i[i] += eps
            params_plus_j = params.copy()
            params_plus_j[j] += eps
            
            # Compute circuit outputs
            output = circuit(params)
            output_i = circuit(params_plus_i)
            output_j = circuit(params_plus_j)
            
            # Compute Fisher Information matrix element
            partial_i = (output_i - output) / eps
            partial_j = (output_j - output) / eps
            qfim[i, j] = np.real(np.dot(partial_i.conj(), partial_j))
    
    return qfim


def measure_entanglement(state: np.ndarray) -> float:
    """
    Measure the entanglement of a two-qubit state using von Neumann entropy.
    
    Args:
        state (np.ndarray): Two-qubit state vector
    
    Returns:
        float: Entanglement measure (von Neumann entropy)
    """
    # Reshape state vector to density matrix
    rho = np.outer(state, state.conj())
    
    # Partial trace over second qubit
    rho_A = np.zeros((2, 2), dtype=complex)
    rho_A[0, 0] = rho[0, 0] + rho[1, 1]
    rho_A[0, 1] = rho[0, 2] + rho[1, 3]
    rho_A[1, 0] = rho[2, 0] + rho[3, 1]
    rho_A[1, 1] = rho[2, 2] + rho[3, 3]
    
    # Compute eigenvalues
    eigenvals = np.linalg.eigvalsh(rho_A)
    eigenvals = eigenvals[eigenvals > 1e-10]  # Remove numerical noise
    
    # Compute von Neumann entropy
    entropy = -np.sum(eigenvals * np.log2(eigenvals))
    return entropy 