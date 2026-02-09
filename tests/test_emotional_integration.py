import unittest
import torch
import numpy as np
from src.components.emotional_integration import EmotionalIntegration

class TestEmotionalIntegration(unittest.TestCase):
    def setUp(self):
        self.integration = EmotionalIntegration()
        self.test_interaction = {
            'valence': 0.5,
            'arousal': 0.3,
            'dominance': 0.4,
            'joy': 0.6,
            'sadness': 0.2,
            'anger': 0.1,
            'fear': 0.3,
            'surprise': 0.4
        }
        self.test_context = {
            'source': 'user',
            'timestamp': '2024-03-20T12:00:00',
            'type': 'conversation'
        }
        
    def test_initialization(self):
        self.assertIsNotNone(self.integration.memory)
        self.assertIsNotNone(self.integration.learning)
        self.assertIsNotNone(self.integration.contagion)
        self.assertIsNotNone(self.integration.regulation)
        self.assertIsNotNone(self.integration.adaptation)
        self.assertEqual(len(self.integration.interaction_history), 0)
        
    def test_process_interaction(self):
        result = self.integration.process_interaction(
            self.test_interaction, self.test_context)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(self.test_interaction))
        self.assertTrue(all(0 <= v <= 1 for v in result.values()))
        self.assertEqual(len(self.integration.interaction_history), 1)
        
    def test_synchronize_emotional_systems(self):
        # Add some test data
        self.integration.process_interaction(self.test_interaction, self.test_context)
        
        # Synchronize systems
        self.integration.synchronize_emotional_systems()
        
        # Check if states are synchronized
        memory_state = self.integration.memory.get_current_state()
        learning_state = self.integration.learning.get_current_state()
        
        self.assertIsNotNone(memory_state)
        self.assertIsNotNone(learning_state)
        
    def test_evolve_emotional_strategy(self):
        # Add some test data
        self.integration.process_interaction(self.test_interaction, self.test_context)
        
        # Evolve strategy
        self.integration.evolve_emotional_strategy(0.8)  # Good performance
        
        # Check if strategy was updated
        current_strategy = self.integration.learning.get_current_strategy()
        self.assertIsNotNone(current_strategy)
        
    def test_get_integration_stats(self):
        # Add some test data
        self.integration.process_interaction(self.test_interaction, self.test_context)
        
        stats = self.integration.get_integration_stats()
        
        self.assertIn('memory_stats', stats)
        self.assertIn('learning_stats', stats)
        self.assertIn('contagion_stats', stats)
        self.assertIn('regulation_stats', stats)
        self.assertIn('adaptation_stats', stats)
        self.assertIn('interaction_count', stats)
        
    def test_save_load_state(self):
        # Add some test data
        self.integration.process_interaction(self.test_interaction, self.test_context)
        
        # Save state
        self.integration.save_integration_state('test_integration_state.pt')
        
        # Create new instance and load state
        new_integration = EmotionalIntegration()
        new_integration.load_integration_state('test_integration_state.pt')
        
        # Compare states
        self.assertEqual(len(self.integration.interaction_history),
                        len(new_integration.interaction_history))
        
    def test_get_emotional_profile(self):
        # Add some test data
        self.integration.process_interaction(self.test_interaction, self.test_context)
        
        profile = self.integration.get_emotional_profile()
        
        self.assertIn('current_state', profile)
        self.assertIn('personality', profile)
        self.assertIn('learning_progress', profile)
        self.assertIn('adaptation_level', profile)
        self.assertIn('regulation_status', profile)
        self.assertIn('contagion_impact', profile)
        
if __name__ == '__main__':
    unittest.main() 