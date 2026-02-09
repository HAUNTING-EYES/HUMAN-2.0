import unittest
import torch
import numpy as np
from src.components.emotional_contagion import EmotionalContagion

class TestEmotionalContagion(unittest.TestCase):
    def setUp(self):
        self.contagion = EmotionalContagion()
        self.test_state = {
            'valence': 0.5,
            'arousal': 0.3,
            'dominance': 0.4,
            'joy': 0.6,
            'sadness': 0.2,
            'anger': 0.1,
            'fear': 0.3,
            'surprise': 0.4
        }
        
    def test_initialization(self):
        self.assertIsNotNone(self.contagion.influence_model)
        self.assertEqual(len(self.contagion.emotional_history), 0)
        self.assertEqual(len(self.contagion.influence_patterns), 0)
        
    def test_propagate_emotion(self):
        target_state = self.test_state.copy()
        target_state['joy'] = 0.2  # Different joy value
        
        new_state = self.contagion.propagate_emotion(self.test_state, target_state)
        
        self.assertIsInstance(new_state, dict)
        self.assertEqual(len(new_state), len(target_state))
        self.assertTrue(all(0 <= v <= 1 for v in new_state.values()))
        
    def test_synchronize_emotions(self):
        states = [
            self.test_state.copy(),
            {k: v * 0.8 for k, v in self.test_state.items()},
            {k: v * 1.2 for k, v in self.test_state.items()}
        ]
        
        synced_states = self.contagion.synchronize_emotions(states)
        
        self.assertEqual(len(synced_states), len(states))
        self.assertTrue(all(0 <= v <= 1 for state in synced_states for v in state.values()))
        
    def test_update_influence_patterns(self):
        self.contagion.update_influence_patterns(self.test_state, self.test_state, True)
        
        pattern_key = tuple(self.test_state.values())
        self.assertIn(pattern_key, self.contagion.influence_patterns)
        self.assertEqual(self.contagion.influence_patterns[pattern_key]['success_count'], 1)
        self.assertEqual(self.contagion.influence_patterns[pattern_key]['total_count'], 1)
        
    def test_get_influence_stats(self):
        # Add some test patterns
        self.contagion.update_influence_patterns(self.test_state, self.test_state, True)
        self.contagion.update_influence_patterns(self.test_state, self.test_state, False)
        
        stats = self.contagion.get_influence_stats()
        
        self.assertIn('total_patterns', stats)
        self.assertIn('avg_success_rate', stats)
        self.assertIn('most_successful_pattern', stats)
        self.assertIn('highest_success_rate', stats)
        
    def test_save_load_state(self):
        # Add some test data
        self.contagion.update_influence_patterns(self.test_state, self.test_state, True)
        
        # Save state
        self.contagion.save_state('test_contagion_state.pt')
        
        # Create new instance and load state
        new_contagion = EmotionalContagion()
        new_contagion.load_state('test_contagion_state.pt')
        
        # Compare states
        self.assertEqual(len(self.contagion.influence_patterns), 
                        len(new_contagion.influence_patterns))
        
if __name__ == '__main__':
    unittest.main() 