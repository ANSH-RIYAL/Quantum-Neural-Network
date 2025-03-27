# Quantum Neural Networks Tutorial

## Table of Contents
1. [Introduction to Quantum Computing](#introduction-to-quantum-computing)
2. [Quantum States and Qubits](#quantum-states-and-qubits)
3. [Quantum Gates and Circuits](#quantum-gates-and-circuits)
4. [Quantum Measurements](#quantum-measurements)
5. [Neural Networks and Quantum Computing](#neural-networks-and-quantum-computing)
6. [Using Our Quantum Neural Network Implementation](#using-our-quantum-neural-network-implementation)

## Introduction to Quantum Computing

Quantum computing is a paradigm that leverages quantum mechanical phenomena to perform computations. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously through superposition.

Key concepts:
- **Superposition**: A qubit can be in multiple states at once
- **Entanglement**: Quantum states can be correlated in ways impossible for classical systems
- **Interference**: Quantum states can interfere with each other, leading to amplification or cancellation of probabilities

## Quantum States and Qubits

### The Qubit
A qubit is the fundamental unit of quantum information. While a classical bit can only be 0 or 1, a qubit can be in a superposition of both states:

\[ |\psi⟩ = \alpha|0⟩ + \beta|1⟩ \]

where:
- |0⟩ and |1⟩ are the basis states (analogous to classical 0 and 1)
- α and β are complex numbers satisfying |α|² + |β|² = 1
- |α|² is the probability of measuring 0
- |β|² is the probability of measuring 1

### Bell States
Bell states are maximally entangled quantum states of two qubits. The most common Bell state is:

\[ |\Phi^+⟩ = \frac{1}{\sqrt{2}}(|00⟩ + |11⟩) \]

This state demonstrates quantum entanglement, where measuring one qubit instantly determines the state of the other.

## Quantum Gates and Circuits

Quantum gates are the building blocks of quantum circuits. Common gates include:

1. **Pauli Gates**:
   - X gate (NOT): Flips the qubit
   - Y gate: Rotation around Y-axis
   - Z gate: Phase flip

2. **Hadamard Gate (H)**:
   Creates superposition:
   \[ H|0⟩ = \frac{1}{\sqrt{2}}(|0⟩ + |1⟩) \]

3. **CNOT Gate**:
   Two-qubit gate that flips the target qubit if the control qubit is |1⟩

## Quantum Measurements

Measuring a quantum state collapses the superposition to a classical state. The measurement outcome is probabilistic based on the quantum state's amplitudes.

Types of measurements:
- **Computational basis**: Measures in |0⟩/|1⟩ basis
- **Pauli measurements**: Measures along X, Y, or Z axes
- **Expectation values**: Average of many measurements

## Neural Networks and Quantum Computing

Quantum Neural Networks (QNNs) combine quantum computing with neural network architectures:

1. **Classical Neural Networks**:
   - Process information using weighted connections
   - Learn through gradient-based optimization
   - Work with classical data

2. **Quantum Neural Networks**:
   - Use quantum circuits as computational units
   - Process quantum and classical data
   - Leverage quantum effects for potentially enhanced learning

3. **Hybrid Quantum-Classical Networks**:
   - Combine classical and quantum layers
   - Pre-process data classically
   - Use quantum operations for specific computations
   - Post-process results classically

## Using Our Quantum Neural Network Implementation

Our implementation provides two main components:

### 1. Quantum Layer
```python
from models.quantum_layers import QuantumLayer

# Create a quantum layer
quantum_layer = QuantumLayer(n_qubits=2, n_layers=2)

# Process data
output = quantum_layer(input_data)
```

The `QuantumLayer` applies quantum operations using:
- Amplitude encoding for input data
- Parameterized quantum circuits
- Measurement of qubit expectation values

### 2. Hybrid Layer
```python
from models.quantum_layers import HybridLayer

# Create a hybrid layer
hybrid_layer = HybridLayer(in_features=3, out_features=2, n_qubits=2)

# Process data
output = hybrid_layer(input_data)
```

The `HybridLayer` combines:
- Classical pre-processing
- Quantum processing
- Classical post-processing

### Interactive Examples

Run `interact.py` to explore the functionality:
```bash
python src/interact.py
```

The script provides interactive demonstrations of:
1. Basic quantum layer operations
2. Hybrid quantum-classical processing
3. Bell state creation and entanglement
4. Training a quantum neural network

Each demonstration includes visualizations and explanations of the quantum processes involved.

### Tips for Using QNNs

1. **Data Preparation**:
   - Normalize input data appropriately
   - Consider the number of qubits needed
   - Ensure data dimensions match the quantum circuit

2. **Model Design**:
   - Start with simple quantum circuits
   - Gradually increase circuit depth
   - Consider hybrid approaches for complex tasks

3. **Training**:
   - Use appropriate learning rates
   - Monitor for barren plateaus
   - Consider quantum-aware optimizers

4. **Evaluation**:
   - Compare with classical baselines
   - Analyze quantum resource requirements
   - Consider the practical advantages/disadvantages 