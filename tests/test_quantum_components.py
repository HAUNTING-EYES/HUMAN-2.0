import unittest
import torch
import numpy as np
from src.components.quantum_inspired_nn import (
    QuantumInspiredNN,
    QuantumLayer,
    SuperpositionTransform,
    EntanglementTransform,
    QuantumMeasurement,
    QuantumTunneling
)

class TestQuantumComponents(unittest.TestCase):
    """Test quantum-inspired neural network components."""
    
    def setUp(self):
        """Set up test environment."""
        self.batch_size = 4
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5
        self.num_qubits = 3
        
        # Initialize quantum components
        self.quantum_nn = QuantumInspiredNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_qubits=self.num_qubits
        )
        
        self.quantum_layer = QuantumLayer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_qubits=self.num_qubits
        )
        
        self.superposition = SuperpositionTransform(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_qubits=self.num_qubits
        )
        
        self.entanglement = EntanglementTransform(
            hidden_size=self.hidden_size,
            num_qubits=self.num_qubits
        )
        
        self.measurement = QuantumMeasurement(
            in_features=self.hidden_size,
            out_features=self.output_size
        )
        
        self.tunneling = QuantumTunneling(
            hidden_size=self.hidden_size,
            num_qubits=self.num_qubits,
            barrier_height=0.5
        )
        
    def test_quantum_nn_forward(self):
        """Test QuantumInspiredNN forward pass."""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.quantum_nn(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        
        # Check if output is normalized
        norms = torch.norm(output, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        
    def test_quantum_layer_forward(self):
        """Test QuantumLayer forward pass."""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.quantum_layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        
        # Check if output is normalized
        norms = torch.norm(output, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        
    def test_superposition_transform(self):
        """Test SuperpositionTransform."""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.superposition(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_qubits, self.hidden_size))
        
        # Check if output is normalized
        norms = torch.norm(output, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        
    def test_entanglement_transform(self):
        """Test EntanglementTransform."""
        x = torch.randn(self.batch_size, self.num_qubits, self.hidden_size)
        output = self.entanglement(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_qubits, self.hidden_size))
        
        # Check if output is normalized
        norms = torch.norm(output, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        
    def test_quantum_measurement(self):
        """Test QuantumMeasurement."""
        x = torch.randn(self.batch_size, self.num_qubits, self.hidden_size)
        output = self.measurement(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        
        # Check if output is normalized
        norms = torch.norm(output, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        
    def test_quantum_tunneling(self):
        """Test QuantumTunneling."""
        x = torch.randn(self.batch_size, self.num_qubits, self.hidden_size)
        output = self.tunneling(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_qubits, self.hidden_size))
        
        # Check if output is normalized
        norms = torch.norm(output, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

if __name__ == '__main__':
    unittest.main() 