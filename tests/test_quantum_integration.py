import unittest
import torch
import torch.nn.functional as F
import numpy as np
from src.components.quantum_inspired_nn import (
    QuantumInspiredNN,
    QuantumLayer,
    SuperpositionTransform,
    EntanglementTransform,
    QuantumMeasurement,
    QuantumTunneling
)

class TestQuantumIntegration(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Define model parameters
        self.input_size = 10
        self.hidden_size = 16
        self.output_size = 8
        self.num_qubits = 4
        self.batch_size = 2
        
        # Initialize the quantum-inspired neural network
        self.model = QuantumInspiredNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_qubits=self.num_qubits
        )
        
    def test_full_quantum_pipeline(self):
        """Test the complete quantum-inspired neural network pipeline."""
        
        # Create sample input
        x = torch.randn(self.batch_size, self.input_size)
        x = F.normalize(x, p=2, dim=-1)  # Normalize input
        
        # Step 1: Forward pass through the entire network
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        
        # Step 2: Test quantum state conversion
        quantum_state = self.model.text_to_quantum_state(x)
        self.assertEqual(
            quantum_state.shape, 
            (self.batch_size, self.num_qubits, self.hidden_size)
        )
        
        # Step 3: Test measurement
        measured = self.model.measure_quantum_states(quantum_state)
        self.assertEqual(measured.shape, (self.batch_size, self.hidden_size))
        
        # Step 4: Test entity extraction
        entities = self.model._extract_entities(quantum_state)
        self.assertTrue(len(entities) > 0)
        self.assertTrue(all(isinstance(e, tuple) and len(e) == 2 for e in entities))
        
        # Step 5: Test sentence extraction
        sentences = self.model._extract_sentences(quantum_state)
        self.assertTrue(len(sentences) > 0)
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 2 for s in sentences))
        
        # Verify all outputs are properly normalized
        self.assertTrue(torch.allclose(torch.norm(output, dim=1), 
                                     torch.ones(self.batch_size), 
                                     atol=1e-6))
        self.assertTrue(torch.allclose(torch.norm(measured, dim=1), 
                                     torch.ones(self.batch_size), 
                                     atol=1e-6))
                                     
    def test_quantum_components(self):
        """Test individual quantum components."""
        
        # Create sample input
        x = torch.randn(self.batch_size, self.input_size)
        x = F.normalize(x, p=2, dim=-1)
        
        # Project to hidden dimension first (since QuantumLayer expects hidden_size input)
        x_hidden = self.model.input_projection(x)  # [batch_size, hidden_size]
        x_hidden = F.normalize(x_hidden, p=2, dim=-1)
        
        # Test QuantumLayer
        quantum_layer = self.model.quantum_layer
        ql_output = quantum_layer(x_hidden)
        self.assertEqual(ql_output.shape, (self.batch_size, self.hidden_size))
        
        # Test SuperpositionTransform
        superposition = quantum_layer.superposition
        # Properly expand input for superposition transform
        x_expanded = x_hidden.unsqueeze(1).expand(-1, self.num_qubits, -1)  # [batch_size, num_qubits, hidden_size]
        sp_output = superposition(x_expanded)
        self.assertEqual(
            sp_output.shape, 
            (self.batch_size, self.num_qubits, self.hidden_size)
        )
        
        # Test EntanglementTransform
        entanglement = quantum_layer.entanglement
        ent_output = entanglement(sp_output)
        self.assertEqual(
            ent_output.shape, 
            (self.batch_size, self.num_qubits, self.hidden_size)
        )
        
        # Test QuantumMeasurement
        measurement = quantum_layer.measurement
        meas_output = measurement(ent_output)
        self.assertEqual(meas_output.shape, (self.batch_size, self.hidden_size))
        
        # Create and test QuantumTunneling
        tunneling = QuantumTunneling(self.hidden_size, self.num_qubits)
        tunnel_output = tunneling(ent_output)
        self.assertEqual(
            tunnel_output.shape, 
            (self.batch_size, self.num_qubits, self.hidden_size)
        )
        
        # Verify all intermediate outputs are properly normalized
        self.assertTrue(torch.allclose(torch.norm(ql_output, dim=1), 
                                     torch.ones(self.batch_size), 
                                     atol=1e-6))
        self.assertTrue(torch.allclose(torch.norm(sp_output, dim=2), 
                                     torch.ones(self.batch_size, self.num_qubits), 
                                     atol=1e-6))
        self.assertTrue(torch.allclose(torch.norm(ent_output, dim=2), 
                                     torch.ones(self.batch_size, self.num_qubits), 
                                     atol=1e-6))
        self.assertTrue(torch.allclose(torch.norm(meas_output, dim=1), 
                                     torch.ones(self.batch_size), 
                                     atol=1e-6))
        self.assertTrue(torch.allclose(torch.norm(tunnel_output, dim=2), 
                                     torch.ones(self.batch_size, self.num_qubits), 
                                     atol=1e-6))

if __name__ == '__main__':
    unittest.main() 