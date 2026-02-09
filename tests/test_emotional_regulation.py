import unittest
import torch
import numpy as np
from src.components.emotional_regulation import EmotionalRegulation

class TestEmotionalRegulation(unittest.TestCase):
    def setUp(self):
        self.regulation = EmotionalRegulation()
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
        self.assertIsNotNone(self.regulation.regulation_model)
        self.assertEqual(len(self.regulation.emotional_history), 0)
        self.assertEqual(len(self.regulation.regulation_patterns), 0)
        
    def test_regulate_emotion(self):
        target_state = self.test_state.copy()
        target_state['joy'] = 0.2  # Different joy value
        
        new_state = self.regulation.regulate_emotion(self.test_state, target_state)
        
        self.assertIsInstance(new_state, dict)
        self.assertEqual(len(new_state), len(target_state))
        self.assertTrue(all(0 <= v <= 1 for v in new_state.values()))
        
    def test_balance_emotions(self):
        states = [
            self.test_state.copy(),
            {k: v * 0.8 for k, v in self.test_state.items()},
            {k: v * 1.2 for k, v in self.test_state.items()}
        ]
        
        balanced_states = self.regulation.balance_emotions(states)
        
        self.assertEqual(len(balanced_states), len(states))
        self.assertTrue(all(0 <= v <= 1 for state in balanced_states for v in state.values()))
        
    def test_update_regulation_patterns(self):
        self.regulation.update_regulation_patterns(self.test_state, self.test_state, True)
        
        pattern_key = tuple(self.test_state.values())
        self.assertIn(pattern_key, self.regulation.regulation_patterns)
        self.assertEqual(self.regulation.regulation_patterns[pattern_key]['success_count'], 1)
        self.assertEqual(self.regulation.regulation_patterns[pattern_key]['total_count'], 1)
        
    def test_get_regulation_stats(self):
        # Add some test patterns
        self.regulation.update_regulation_patterns(self.test_state, self.test_state, True)
        self.regulation.update_regulation_patterns(self.test_state, self.test_state, False)
        
        stats = self.regulation.get_regulation_stats()
        
        self.assertIn('total_patterns', stats)
        self.assertIn('avg_success_rate', stats)
        self.assertIn('most_successful_pattern', stats)
        self.assertIn('highest_success_rate', stats)
        
    def test_save_load_state(self):
        # Add some test data
        self.regulation.update_regulation_patterns(self.test_state, self.test_state, True)
        
        # Save state
        self.regulation.save_state('test_regulation_state.pt')
        
        # Create new instance and load state
        new_regulation = EmotionalRegulation()
        new_regulation.load_state('test_regulation_state.pt')
        
        # Compare states
        self.assertEqual(len(self.regulation.regulation_patterns), 
                        len(new_regulation.regulation_patterns))
        
if __name__ == '__main__':
    unittest.main() 