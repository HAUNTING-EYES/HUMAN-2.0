import unittest
import torch
import numpy as np
from src.components.emotional_adaptation import EmotionalAdaptation

class TestEmotionalAdaptation(unittest.TestCase):
    def setUp(self):
        self.adaptation = EmotionalAdaptation()
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
        self.assertIsNotNone(self.adaptation.adaptation_model)
        self.assertEqual(len(self.adaptation.emotional_history), 0)
        self.assertEqual(len(self.adaptation.adaptation_patterns), 0)
        
    def test_adapt_emotion(self):
        experience = self.test_state.copy()
        experience['joy'] = 0.8  # Different joy value
        
        new_state = self.adaptation.adapt_emotion(self.test_state, experience)
        
        self.assertIsInstance(new_state, dict)
        self.assertEqual(len(new_state), len(self.test_state))
        self.assertTrue(all(0 <= v <= 1 for v in new_state.values()))
        
    def test_evolve_strategy(self):
        current_strategy = self.test_state.copy()
        performance = 0.8  # Good performance
        
        new_strategy = self.adaptation.evolve_strategy(current_strategy, performance)
        
        self.assertIsInstance(new_strategy, dict)
        self.assertEqual(len(new_strategy), len(current_strategy))
        self.assertTrue(all(0 <= v <= 1 for v in new_strategy.values()))
        
    def test_update_adaptation_patterns(self):
        experience = self.test_state.copy()
        experience['joy'] = 0.8
        
        self.adaptation.update_adaptation_patterns(self.test_state, experience, True)
        
        pattern_key = tuple(self.test_state.values())
        self.assertIn(pattern_key, self.adaptation.adaptation_patterns)
        self.assertEqual(self.adaptation.adaptation_patterns[pattern_key]['success_count'], 1)
        self.assertEqual(self.adaptation.adaptation_patterns[pattern_key]['total_count'], 1)
        self.assertEqual(len(self.adaptation.adaptation_patterns[pattern_key]['experiences']), 1)
        
    def test_get_adaptation_stats(self):
        # Add some test patterns
        experience = self.test_state.copy()
        experience['joy'] = 0.8
        
        self.adaptation.update_adaptation_patterns(self.test_state, experience, True)
        self.adaptation.update_adaptation_patterns(self.test_state, experience, False)
        
        stats = self.adaptation.get_adaptation_stats()
        
        self.assertIn('total_patterns', stats)
        self.assertIn('avg_success_rate', stats)
        self.assertIn('most_successful_pattern', stats)
        self.assertIn('highest_success_rate', stats)
        self.assertIn('avg_experience_count', stats)
        
    def test_save_load_state(self):
        # Add some test data
        experience = self.test_state.copy()
        experience['joy'] = 0.8
        self.adaptation.update_adaptation_patterns(self.test_state, experience, True)
        
        # Save state
        self.adaptation.save_state('test_adaptation_state.pt')
        
        # Create new instance and load state
        new_adaptation = EmotionalAdaptation()
        new_adaptation.load_state('test_adaptation_state.pt')
        
        # Compare states
        self.assertEqual(len(self.adaptation.adaptation_patterns), 
                        len(new_adaptation.adaptation_patterns))
        
if __name__ == '__main__':
    unittest.main() 