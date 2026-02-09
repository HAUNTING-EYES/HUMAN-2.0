#!/usr/bin/env python3
"""Comprehensive test script for HUMAN 2.0 components."""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Import components
from components.brainstem import Brainstem
from components.emotional_memory import EmotionalMemory
from components.quantum_inspired_nn import QuantumLayer, QuantumInspiredNN
from components.quantum_measurement import QuantumMeasurement
from components.quantum_tunneling import QuantumTunneling

class ComponentTester:
    """Test suite for HUMAN 2.0 components."""
    
    def __init__(self):
        """Initialize test environment."""
        self.base_dir = Path.home() / '.human2'
        self.data_dir = self.base_dir / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ComponentTester')
        
    def test_quantum_components(self):
        """Test quantum-inspired components."""
        print("\n=== Testing Quantum Components ===")
        
        try:
            # Test Quantum Layer
            print("\nTesting Quantum Layer...")
            quantum_layer = QuantumLayer()
            state = quantum_layer.initialize_state()
            print(f"Initialized quantum state: {state}")
            
            # Test Quantum-Inspired Neural Network
            print("\nTesting Quantum-Inspired Neural Network...")
            qnn = QuantumInspiredNN(input_size=10, hidden_size=20, output_size=5)
            test_input = [0.1] * 10
            output = qnn.forward(test_input)
            print(f"QNN output shape: {output.shape}")
            
            # Test Quantum Measurement
            print("\nTesting Quantum Measurement...")
            measurement = QuantumMeasurement()
            result = measurement.measure(state)
            print(f"Measurement result: {result}")
            
            # Test Quantum Tunneling
            print("\nTesting Quantum Tunneling...")
            tunneling = QuantumTunneling()
            tunneled_state = tunneling.tunnel(state)
            print(f"Tunneled state: {tunneled_state}")
            
            print("Quantum components test completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing quantum components: {str(e)}")
            return False
            
    def test_emotional_memory(self):
        """Test emotional memory component."""
        print("\n=== Testing Emotional Memory ===")
        
        try:
            # Initialize emotional memory
            emotional_memory = EmotionalMemory(str(self.data_dir / 'emotional_memory'))
            
            # Test basic emotion processing
            print("\nTesting emotion processing...")
            test_inputs = [
                "I am feeling very happy today!",
                "This makes me sad and disappointed.",
                "I'm really angry about this situation.",
                "I'm scared of what might happen next.",
                "I'm surprised by this unexpected turn of events."
            ]
            
            for text in test_inputs:
                result = emotional_memory.process_interaction(text)
                print(f"\nInput: {text}")
                print(f"Emotional state: {result['emotional_state']}")
                print(f"Response: {result['emotional_response']}")
                
            # Test empathy simulation
            print("\nTesting empathy simulation...")
            emotions = ['happy', 'sad', 'angry', 'fear', 'surprise']
            for emotion in emotions:
                response = emotional_memory.simulate_empathy(emotion)
                print(f"\nEmpathy for {emotion}: {response}")
                
            # Test emotional emergence
            print("\nTesting emotional emergence...")
            emergence = emotional_memory.get_emotional_emergence()
            if emergence['emergence_detected']:
                print(f"Emergence type: {emergence['emergence_type']}")
                print(f"Stability: {emergence['emotional_stability']:.2f}")
                print(f"Adaptation: {emergence['emotional_adaptation']:.2f}")
                print(f"Resilience: {emergence['emotional_resilience']:.2f}")
                
            # Test personality traits
            print("\nTesting personality traits...")
            print(f"Personality traits: {emotional_memory.personality_traits}")
            print(f"Empathy level: {emotional_memory._calculate_empathy_level():.2f}")
            
            print("Emotional memory test completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing emotional memory: {str(e)}")
            return False
            
    def test_brainstem(self):
        """Test brainstem component."""
        print("\n=== Testing Brainstem ===")
        
        try:
            # Initialize brainstem
            brainstem = Brainstem(str(self.data_dir))
            
            # Test task processing
            print("\nTesting task processing...")
            test_tasks = [
                "Analyze this code for bugs",
                "Learn about quantum computing",
                "Process emotional input",
                "Make a decision about resource allocation"
            ]
            
            for task in test_tasks:
                result = brainstem.process_task(task)
                print(f"\nTask: {task}")
                print(f"Result: {result}")
                
            # Test state management
            print("\nTesting state management...")
            state = brainstem.get_state()
            print(f"Current state: {state}")
            
            # Test resource allocation
            print("\nTesting resource allocation...")
            resources = brainstem.allocate_resources()
            print(f"Allocated resources: {resources}")
            
            print("Brainstem test completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing brainstem: {str(e)}")
            return False
            
    def run_all_tests(self):
        """Run all component tests."""
        print("Starting comprehensive component tests...")
        
        results = {
            'quantum_components': self.test_quantum_components(),
            'emotional_memory': self.test_emotional_memory(),
            'brainstem': self.test_brainstem()
        }
        
        print("\n=== Test Results Summary ===")
        for component, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{component}: {status}")
            
        # Overall status
        all_passed = all(results.values())
        print(f"\nOverall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

if __name__ == "__main__":
    tester = ComponentTester()
    tester.run_all_tests() 