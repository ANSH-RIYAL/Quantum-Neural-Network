"""
Unit tests for quantum circuit implementations.
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.quantum_layers import QuantumLayer, HybridLayer
from src.utils.quantum_utils import normalize_data, create_bell_state, measure_entanglement


class TestQuantumCircuits(unittest.TestCase):
    """Test cases for quantum circuit implementations."""
    
    def setUp(self):
        """Set up test cases."""
        self.quantum_layer = QuantumLayer(n_qubits=2, n_layers=2)
        self.hybrid_layer = HybridLayer(in_features=3, out_features=2, n_qubits=2)
    
    def test_quantum_layer_output_shape(self):
        """Test if quantum layer produces correct output shape."""
        batch_size = 4
        input_size = 4  # 2^2 for 2 qubits
        x = torch.randn(batch_size, input_size, dtype=torch.float32)
        
        output = self.quantum_layer(x)
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_hybrid_layer_output_shape(self):
        """Test if hybrid layer produces correct output shape."""
        batch_size = 4
        input_size = 3
        x = torch.randn(batch_size, input_size, dtype=torch.float32)
        
        output = self.hybrid_layer(x)
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_data_normalization(self):
        """Test if data normalization works correctly."""
        data = np.random.randn(10, 4)
        normalized_data = normalize_data(data)
        
        # Check if vectors are normalized (L2 norm ≈ 1)
        norms = np.linalg.norm(normalized_data, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=6)
    
    def test_bell_state_entanglement(self):
        """Test if Bell state creation produces maximally entangled state."""
        state = create_bell_state()
        entanglement = measure_entanglement(state)
        
        # Bell state should have maximum entanglement (1 bit)
        self.assertAlmostEqual(entanglement, 1.0, places=6)
    
    def test_quantum_layer_gradient(self):
        """Test if quantum layer is differentiable."""
        # Create a simple optimization task
        batch_size = 2
        input_size = 4  # 2^2 for 2 qubits
        x = torch.randn(batch_size, input_size, dtype=torch.float32, requires_grad=True)
        target = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
        
        # Define optimizer
        optimizer = torch.optim.Adam(self.quantum_layer.parameters(), lr=0.1)
        
        # Single optimization step
        optimizer.zero_grad()
        output = self.quantum_layer(x)
        loss = torch.mean((output - target)**2)
        
        try:
            loss.backward()
            # Check if gradients were computed
            has_gradient = self.quantum_layer.weights.grad is not None
            grad_is_nonzero = torch.any(self.quantum_layer.weights.grad != 0)
            has_gradient = has_gradient and grad_is_nonzero
        except Exception:
            has_gradient = False
        
        self.assertTrue(has_gradient)
    
    def test_invalid_input_size(self):
        """Test if quantum layer raises error for invalid input size."""
        x = torch.randn(4, 3)  # Wrong input size for 2 qubits
        
        with self.assertRaises(ValueError):
            _ = self.quantum_layer(x)


if __name__ == '__main__':
    unittest.main() 