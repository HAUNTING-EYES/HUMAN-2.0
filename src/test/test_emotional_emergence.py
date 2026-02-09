import unittest
import numpy as np
from datetime import datetime
from src.components.emotional_integration import EmotionalIntegration

class TestEmotionalEmergence(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.integration = EmotionalIntegration(
            memory_size=100,
            learning_rate=0.01,
            influence_threshold=0.3,
            balance_threshold=0.5,
            adaptation_rate=0.1
        )
        
        # Define test states
        self.neutral_state = {
            'valence': 0.0,
            'arousal': 0.5,
            'dominance': 0.5,
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0
        }
        
        self.happy_state = {
            'valence': 0.8,
            'arousal': 0.7,
            'dominance': 0.6,
            'joy': 0.9,
            'sadness': 0.1,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.3
        }
        
        self.sad_state = {
            'valence': -0.7,
            'arousal': 0.3,
            'dominance': 0.3,
            'joy': 0.1,
            'sadness': 0.8,
            'anger': 0.2,
            'fear': 0.4,
            'surprise': 0.1
        }
        
    def test_resonance_detection(self):
        """Test emotional resonance detection."""
        # Test strong resonance
        patterns = self.integration._detect_emergence_patterns(
            self.happy_state,
            self.happy_state,
            {'type': 'test'}
        )
        resonance_patterns = [p for p in patterns if p['type'] == 'resonance']
        self.assertTrue(len(resonance_patterns) > 0)
        self.assertGreater(resonance_patterns[0]['strength'], 0.7)
        
        # Test weak resonance
        patterns = self.integration._detect_emergence_patterns(
            self.happy_state,
            self.sad_state,
            {'type': 'test'}
        )
        resonance_patterns = [p for p in patterns if p['type'] == 'resonance']
        self.assertTrue(len(resonance_patterns) == 0)
        
    def test_contagion_detection(self):
        """Test emotional contagion detection."""
        # Create intensifying state
        intensified_state = self.happy_state.copy()
        for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise']:
            if intensified_state[emotion] > 0.5:
                intensified_state[emotion] = min(1.0, intensified_state[emotion] + 0.2)
        
        # Test strong contagion
        patterns = self.integration._detect_emergence_patterns(
            self.happy_state,
            intensified_state,
            {'type': 'test'}
        )
        contagion_patterns = [p for p in patterns if p['type'] == 'contagion']
        self.assertTrue(len(contagion_patterns) > 0)
        self.assertGreater(contagion_patterns[0]['strength'], 0.6)
        
        # Test weak contagion
        patterns = self.integration._detect_emergence_patterns(
            self.happy_state,
            self.neutral_state,
            {'type': 'test'}
        )
        contagion_patterns = [p for p in patterns if p['type'] == 'contagion']
        self.assertTrue(len(contagion_patterns) == 0)
        
    def test_regulation_detection(self):
        """Test emotional regulation detection."""
        # Create regulated state
        regulated_state = self.happy_state.copy()
        for dim in ['valence', 'arousal', 'dominance']:
            if regulated_state[dim] > 0.5:
                regulated_state[dim] = 0.5 + (regulated_state[dim] - 0.5) * 0.5
        
        # Test strong regulation
        patterns = self.integration._detect_emergence_patterns(
            self.happy_state,
            regulated_state,
            {'type': 'test'}
        )
        regulation_patterns = [p for p in patterns if p['type'] == 'regulation']
        self.assertTrue(len(regulation_patterns) > 0)
        self.assertGreater(regulation_patterns[0]['strength'], 0.5)
        
        # Test weak regulation
        patterns = self.integration._detect_emergence_patterns(
            self.neutral_state,
            self.neutral_state,
            {'type': 'test'}
        )
        regulation_patterns = [p for p in patterns if p['type'] == 'regulation']
        self.assertTrue(len(regulation_patterns) == 0)
        
    def test_adaptation_detection(self):
        """Test emotional adaptation detection."""
        # Create adapted state
        adapted_state = self.happy_state.copy()
        context_valence = -0.5
        adaptation_rate = 0.3
        adapted_state['valence'] = self.happy_state['valence'] + (context_valence - self.happy_state['valence']) * adaptation_rate
        
        # Test strong adaptation
        patterns = self.integration._detect_emergence_patterns(
            self.happy_state,
            adapted_state,
            {'type': 'test', 'valence': context_valence}
        )
        adaptation_patterns = [p for p in patterns if p['type'] == 'adaptation']
        self.assertTrue(len(adaptation_patterns) > 0)
        self.assertGreater(adaptation_patterns[0]['strength'], 0.4)
        
        # Test weak adaptation
        patterns = self.integration._detect_emergence_patterns(
            self.neutral_state,
            self.neutral_state,
            {'type': 'test'}
        )
        adaptation_patterns = [p for p in patterns if p['type'] == 'adaptation']
        self.assertTrue(len(adaptation_patterns) == 0)
        
    def test_resonance_effect(self):
        """Test applying resonance effect."""
        # Test resonance effect on happy state
        resonated_state = self.integration._apply_resonance_effect(
            self.happy_state,
            0.8  # Strong resonance
        )
        self.assertGreater(resonated_state['joy'], self.happy_state['joy'])
        self.assertLess(resonated_state['sadness'], self.happy_state['sadness'])
        
        # Test resonance effect on neutral state
        resonated_state = self.integration._apply_resonance_effect(
            self.neutral_state,
            0.8  # Strong resonance
        )
        self.assertEqual(resonated_state, self.neutral_state)
        
    def test_contagion_effect(self):
        """Test applying contagion effect."""
        # Test contagion effect on happy state
        contagion_state = self.integration._apply_contagion_effect(
            self.happy_state,
            0.8  # Strong contagion
        )
        self.assertGreater(contagion_state['joy'], self.happy_state['joy'])
        self.assertLess(contagion_state['sadness'], self.happy_state['sadness'])
        
        # Test contagion effect on neutral state
        contagion_state = self.integration._apply_contagion_effect(
            self.neutral_state,
            0.8  # Strong contagion
        )
        self.assertEqual(contagion_state, self.neutral_state)
        
    def test_regulation_effect(self):
        """Test applying regulation effect."""
        # Test regulation effect on extreme state
        regulated_state = self.integration._apply_regulation_effect(
            self.happy_state,
            0.8  # Strong regulation
        )
        self.assertLess(regulated_state['valence'], self.happy_state['valence'])
        self.assertLess(regulated_state['arousal'], self.happy_state['arousal'])
        
        # Test regulation effect on balanced state
        regulated_state = self.integration._apply_regulation_effect(
            self.neutral_state,
            0.8  # Strong regulation
        )
        self.assertAlmostEqual(regulated_state['valence'], self.neutral_state['valence'], places=2)
        self.assertAlmostEqual(regulated_state['arousal'], self.neutral_state['arousal'], places=2)
        
    def test_adaptation_effect(self):
        """Test applying adaptation effect."""
        context = {
            'valence': -0.5,
            'arousal': 0.3
        }
        
        # Test adaptation effect with context
        adapted_state = self.integration._apply_adaptation_effect(
            self.happy_state,
            0.8,  # Strong adaptation
            context
        )
        self.assertLess(adapted_state['valence'], self.happy_state['valence'])
        self.assertLess(adapted_state['arousal'], self.happy_state['arousal'])
        
        # Test adaptation effect without context
        adapted_state = self.integration._apply_adaptation_effect(
            self.happy_state,
            0.8  # Strong adaptation
        )
        self.assertNotEqual(adapted_state, self.happy_state)
        
    def test_emergence_integration(self):
        """Test full emergence pattern integration."""
        # Process interaction that should trigger multiple patterns
        result = self.integration.process_interaction(
            interaction={
                'type': 'test',
                'valence': 0.8,
                'arousal': 0.7
            },
            context={
                'valence': 0.7,
                'arousal': 0.6
            }
        )
        
        # Verify patterns were detected and applied
        self.assertTrue(result['success'])
        self.assertTrue(len(result['patterns']) > 0)
        self.assertNotEqual(result['state'], self.integration.current_state)
        
        # Verify state changes are within valid ranges
        for dim in ['valence', 'arousal', 'dominance']:
            if dim == 'valence':
                self.assertGreaterEqual(result['state'][dim], -1.0)
                self.assertLessEqual(result['state'][dim], 1.0)
            else:
                self.assertGreaterEqual(result['state'][dim], 0.0)
                self.assertLessEqual(result['state'][dim], 1.0)
        
    def test_emergence_error_handling(self):
        """Test error handling in emergence pattern processing."""
        # Test with invalid state
        invalid_state = {'invalid': 'state'}
        patterns = self.integration._detect_emergence_patterns(
            invalid_state,
            invalid_state,
            {'type': 'test'}
        )
        self.assertEqual(patterns, [])
        
        # Test with invalid pattern type
        emerged_state = self.integration._apply_emergence_effects(
            self.neutral_state,
            [{'type': 'invalid', 'strength': 0.8}]
        )
        self.assertEqual(emerged_state, self.neutral_state)
        
        # Test with None values
        result = self.integration.process_interaction(
            interaction=None,
            context=None
        )
        self.assertFalse(result['success'])
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main() 