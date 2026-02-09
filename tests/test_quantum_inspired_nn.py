import unittest
import torch
import torch.nn.functional as F
from src.components.quantum_inspired_nn import (
    QuantumInspiredNN,
    QuantumLayer,
    SuperpositionTransform,
    EntanglementTransform,
    QuantumMeasurement,
    QuantumTunneling
)

class TestQuantumInspiredNN(unittest.TestCase):
    def setUp(self):
        self.input_size = 32
        self.hidden_size = 64
        self.output_size = 16
        self.num_qubits = 4
        self.batch_size = 8
        
        self.model = QuantumInspiredNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_qubits=self.num_qubits
        )
        self.input_tensor = torch.randn(self.batch_size, self.input_size)
        
    def test_initialization(self):
        self.assertIsInstance(self.model.input_projection, torch.nn.Linear)
        self.assertIsInstance(self.model.quantum_layer, QuantumLayer)
        self.assertIsInstance(self.model.output_projection, torch.nn.Linear)
        
    def test_forward_pass(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        # Check normalization
        norms = torch.norm(output, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

class TestQuantumLayer(unittest.TestCase):
    def setUp(self):
        self.input_size = 32
        self.hidden_size = 64
        self.num_qubits = 4
        self.batch_size = 8
        
        self.layer = QuantumLayer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_qubits=self.num_qubits
        )
        self.input_tensor = torch.randn(self.batch_size, self.input_size)
        
    def test_initialization(self):
        self.assertIsInstance(self.layer.superposition, SuperpositionTransform)
        self.assertIsInstance(self.layer.entanglement, EntanglementTransform)
        self.assertIsInstance(self.layer.measurement, QuantumMeasurement)
        
    def test_forward_pass(self):
        output = self.layer(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        # Check output is valid
        self.assertTrue(torch.all(torch.isfinite(output)))

class TestSuperpositionTransform(unittest.TestCase):
    def setUp(self):
        self.input_size = 32
        self.hidden_size = 64
        self.num_qubits = 4
        self.batch_size = 8
        
        self.transform = SuperpositionTransform(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_qubits=self.num_qubits
        )
        
    def test_initialization(self):
        self.assertEqual(self.transform.input_size, self.input_size)
        self.assertEqual(self.transform.hidden_size, self.hidden_size)
        self.assertEqual(self.transform.num_qubits, self.num_qubits)
        self.assertEqual(self.transform.weight.shape, (self.input_size, self.hidden_size))
        
    def test_forward_pass(self):
        x = torch.randn(self.batch_size, self.num_qubits, self.input_size)
        output = self.transform(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_qubits, self.hidden_size))
        # Check normalization
        norms = torch.norm(output, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

class TestEntanglementTransform(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 64
        self.num_qubits = 4
        self.batch_size = 8
        
        self.transform = EntanglementTransform(
            hidden_size=self.hidden_size,
            num_qubits=self.num_qubits
        )
        
    def test_initialization(self):
        self.assertEqual(self.transform.hidden_size, self.hidden_size)
        self.assertEqual(self.transform.num_qubits, self.num_qubits)
        self.assertEqual(self.transform.entanglement_weights.shape, (self.num_qubits, self.num_qubits, self.hidden_size))
        
    def test_forward_pass(self):
        x = torch.randn(self.batch_size, self.num_qubits, self.hidden_size)
        output = self.transform(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_qubits, self.hidden_size))
        # Check normalization
        norms = torch.norm(output, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

class TestQuantumMeasurement(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 64
        self.output_size = 32
        self.batch_size = 8
        self.num_qubits = 4
        
        self.measurement = QuantumMeasurement(
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )
        
    def test_initialization(self):
        self.assertEqual(self.measurement.hidden_size, self.hidden_size)
        self.assertEqual(self.measurement.output_size, self.output_size)
        self.assertEqual(self.measurement.measurement_matrix.shape, (self.hidden_size, self.output_size))
        
    def test_forward_pass(self):
        x = torch.randn(self.batch_size, self.num_qubits, self.hidden_size)
        output = self.measurement(x)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        # Check normalization
        norms = torch.norm(output, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

class TestQuantumTunneling(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 64
        self.num_qubits = 4
        self.batch_size = 8
        
        self.tunneling = QuantumTunneling(
            hidden_size=self.hidden_size,
            num_qubits=self.num_qubits,
            barrier_height=1.0
        )
        
    def test_initialization(self):
        self.assertEqual(self.tunneling.hidden_size, self.hidden_size)
        self.assertEqual(self.tunneling.num_qubits, self.num_qubits)
        self.assertEqual(self.tunneling.tunneling_weights.shape, (self.num_qubits, self.hidden_size))
        
    def test_forward_pass(self):
        x = torch.randn(self.batch_size, self.num_qubits, self.hidden_size)
        output = self.tunneling(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_qubits, self.hidden_size))
        # Check normalization
        norms = torch.norm(output, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

if __name__ == '__main__':
    unittest.main() 