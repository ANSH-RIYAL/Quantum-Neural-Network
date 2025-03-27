# From Neural Networks to Quantum Neural Networks

A comprehensive guide and implementation for understanding the transition from classical Neural Networks to Quantum Neural Networks.

Author: Ansh Riyal

## Overview

This repository is designed to help you understand how quantum computing concepts can be integrated with neural networks. I've created this as a learning resource for anyone interested in the intersection of quantum computing and machine learning.

## Repository Structure

```
.
├── src/                      # Source code directory
│   ├── models/              # Core model implementations
│   │   └── quantum_layers.py    # Quantum and hybrid layer implementations
│   ├── utils/               # Utility functions
│   │   └── quantum_utils.py     # Quantum computing utilities
│   ├── experiments/         # Comparative studies
│   │   └── comparative_study.py # QNN vs Classical NN/CNN comparisons
│   └── interact.py          # Interactive demonstrations
├── tests/                   # Unit tests
│   └── test_quantum_circuits.py # Tests for quantum implementations
├── data/                    # Dataset storage
├── Tutorial.md              # Comprehensive quantum computing tutorial
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## File Descriptions

### Core Implementation Files
- `src/models/quantum_layers.py`: Contains the core quantum neural network implementations:
  - `QuantumLayer`: Pure quantum neural network layer
  - `HybridLayer`: Hybrid quantum-classical layer

- `src/utils/quantum_utils.py`: Utility functions for quantum computing:
  - Bell state creation
  - Entanglement measurement
  - Data normalization

### Experiment and Demo Files
- `src/experiments/comparative_study.py`: Comparative analysis of:
  - QNN vs Classical NN on MNIST
  - QNN vs CNN on Fashion-MNIST
  - Performance metrics and visualizations

- `src/interact.py`: Interactive demonstration script with:
  - Quantum layer demonstrations
  - Hybrid layer examples
  - Bell state visualization
  - Training examples

### Documentation
- `Tutorial.md`: Comprehensive tutorial covering:
  - Quantum computing basics
  - Neural network fundamentals
  - Integration of quantum and classical computing
  - Practical examples and use cases

### Testing
- `tests/test_quantum_circuits.py`: Unit tests for:
  - Quantum layer functionality
  - Hybrid layer operations
  - Utility functions

## Installation

1. Create a virtual environment:
```bash
python -m venv qnn
source qnn/bin/activate  # On Windows: qnn\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. Run the interactive demo:
```bash
python src/interact.py
```

2. Compare classical and quantum models:
```bash
python src/experiments/comparative_study.py
```

## Learning Path

1. Start with `Tutorial.md` for theoretical understanding
2. Explore `interact.py` for hands-on demonstrations
3. Study the model implementations in `src/models/`
4. Run the comparative experiments to understand trade-offs

## Key Components

### Quantum Layer
The `QuantumLayer` class implements a pure quantum neural network layer:
```python
from models.quantum_layers import QuantumLayer

# Create a quantum layer with 4 qubits and 2 layers
quantum_layer = QuantumLayer(n_qubits=4, n_layers=2)
```

### Hybrid Layer
The `HybridLayer` combines classical and quantum processing:
```python
from models.quantum_layers import HybridLayer

# Create a hybrid layer
hybrid_layer = HybridLayer(in_features=10, out_features=2, n_qubits=4)
```

## Experiments and Results

I've included two main comparative studies:
1. QNN vs Classical NN on MNIST (digits 0 and 1)
2. QNN vs CNN on Fashion-MNIST (t-shirts vs shirts)

The experiments demonstrate:
- Training behavior differences
- Performance metrics comparison
- Resource utilization
- Practical considerations

## Contributing

Feel free to contribute by:
1. Opening issues for questions or bugs
2. Submitting pull requests for improvements
3. Adding more experiments or tutorials

## License

MIT License - feel free to use this for learning and development!

## Contact

- Author: Ansh Riyal
- Email: ansh.riyal@nyu.edu

## Acknowledgments

Special thanks to:
- PennyLane team for their quantum computing framework
- PyTorch team for their deep learning framework
- The quantum computing community for their research and insights
