#!/usr/bin/env python3
"""Advanced test script for the EmotionalMemory component."""

import unittest
import torch
from pathlib import Path
import tempfile
import shutil
from src.components.emotional_memory import EmotionalMemory
from datetime import datetime
import numpy as np
import time

class TestEmotionalMemoryAdvanced(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.emotional_memory = EmotionalMemory(base_dir=self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def test_edge_case_emotional_states(self):
        """Test edge cases in emotional state handling."""
        # Test extreme values
        extreme_state = {
            'valence': 1.0,
            'arousal': 1.0,
            'dominance': 1.0,
            'novelty': 1.0,
            'complexity': 1.0,
            'intensity': 1.0,
            'stability': 1.0,
            'coherence': 1.0
        }
        result = self.emotional_memory.update_emotional_state(extreme_state)
        self.assertTrue(all(0 <= v <= 1 for v in result['state'].values() if isinstance(v, (int, float))))
        
        # Test negative values
        negative_state = {
            'valence': -1.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'novelty': 0.0,
            'complexity': 0.0,
            'intensity': 0.0,
            'stability': 0.0,
            'coherence': 0.0
        }
        result = self.emotional_memory.update_emotional_state(negative_state)
        self.assertTrue(all(-1 <= v <= 1 for v in result['state'].values() if isinstance(v, (int, float))))
        
    def test_emotional_learning(self):
        """Test emotional learning capabilities."""
        # Initial emotional state
        initial_state = self.emotional_memory.get_personality_profile()
        
        # Simulate learning from positive experiences
        for _ in range(5):
            self.emotional_memory.process_interaction("I am feeling very happy and successful!")
            
        # Simulate learning from negative experiences
        for _ in range(5):
            self.emotional_memory.process_interaction("I am feeling sad and disappointed.")
            
        # Get updated profile
        updated_state = self.emotional_memory.get_personality_profile()
        
        # Verify learning effects
        self.assertNotEqual(initial_state['traits'], updated_state['traits'])
        self.assertNotEqual(initial_state['adaptability'], updated_state['adaptability'])
        self.assertNotEqual(initial_state['resilience'], updated_state['resilience'])
        
    def test_emotional_pattern_recognition(self):
        """Test recognition of emotional patterns."""
        # Create a pattern of emotions
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise']
        intensities = [0.8, 0.6, 0.7, 0.5, 0.9]
        
        for emotion, intensity in zip(emotions, intensities):
            self.emotional_memory.process_emotion(emotion, intensity, f"Testing {emotion}")
            
        # Detect patterns
        patterns = self.emotional_memory.detect_emotional_patterns()
        
        # Verify pattern detection
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Verify pattern structure
        for pattern in patterns:
            self.assertIn('type', pattern)
            self.assertIn('confidence', pattern)
            self.assertIn('description', pattern)
            
    def test_emotional_adaptation(self):
        """Test emotional adaptation mechanisms."""
        # Initial emotional state
        initial_state = self.emotional_memory.emotional_state.copy()
        
        # Simulate repeated emotional stimuli
        for _ in range(10):
            self.emotional_memory.process_emotion('joy', 0.8, "Repeated positive stimulus")
            
        # Get adapted state
        adapted_state = self.emotional_memory.emotional_state
        
        # Verify adaptation
        self.assertNotEqual(initial_state, adapted_state)
        
        # Test adaptation to negative stimuli
        for _ in range(10):
            self.emotional_memory.process_emotion('sadness', 0.8, "Repeated negative stimulus")
            
        # Get final adapted state
        final_state = self.emotional_memory.emotional_state
        
        # Verify adaptation to negative stimuli
        self.assertNotEqual(adapted_state, final_state)
        
    def test_emotional_resilience(self):
        """Test emotional resilience development."""
        # Initial resilience
        initial_profile = self.emotional_memory.get_personality_profile()
        initial_resilience = initial_profile['resilience']
        
        # Simulate emotional challenges
        challenges = [
            ('joy', 0.9, "Extreme happiness"),
            ('sadness', 0.9, "Extreme sadness"),
            ('anger', 0.9, "Extreme anger"),
            ('fear', 0.9, "Extreme fear")
        ]
        
        for emotion, intensity, context in challenges:
            self.emotional_memory.process_emotion(emotion, intensity, context)
            
        # Get updated profile
        updated_profile = self.emotional_memory.get_personality_profile()
        
        # Verify resilience development
        self.assertGreater(updated_profile['resilience'], initial_resilience)
        
    def test_emotional_memory_consolidation(self):
        """Test emotional memory consolidation."""
        # Add multiple emotional experiences
        experiences = [
            ('joy', 0.8, "Happy moment"),
            ('sadness', 0.6, "Sad moment"),
            ('anger', 0.7, "Angry moment"),
            ('fear', 0.5, "Scary moment"),
            ('surprise', 0.9, "Surprising moment")
        ]
        
        for emotion, intensity, context in experiences:
            self.emotional_memory.process_emotion(emotion, intensity, context)
            
        # Force memory consolidation
        self.emotional_memory._consolidate_memories()
        
        # Verify consolidation
        self.assertGreater(len(self.emotional_memory.long_term_memory), 0)
        
        # Verify memory structure
        for memory in self.emotional_memory.long_term_memory:
            self.assertIn('emotion', memory)
            self.assertIn('intensity', memory)
            self.assertIn('context', memory)
            self.assertIn('timestamp', memory)
            self.assertIn('importance', memory)
            
    def test_emotional_contagion(self):
        """Test emotional contagion effects."""
        # Initial emotional state
        initial_state = self.emotional_memory.emotional_state.copy()
        
        # Create emotional context
        context = {
            'valence': 0.8,
            'arousal': 0.7,
            'dominance': 0.6,
            'novelty': 0.5,
            'complexity': 0.4,
            'intensity': 0.9,
            'stability': 0.3,
            'coherence': 0.7
        }
        
        # Apply emotional contagion
        result = self.emotional_memory.process_interaction(
            "I am feeling very happy!",
            context=context
        )
        
        # Verify contagion effects
        self.assertNotEqual(initial_state, result['emotional_state'])
        
        # Test personality influence on contagion
        personality_profile = self.emotional_memory.get_personality_profile()
        self.assertIn('extraversion', personality_profile['traits'])
        self.assertIn('neuroticism', personality_profile['traits'])
        
    def test_emotional_emergence_stability(self):
        """Test stability of emotional emergence patterns."""
        # Create emotional history
        for _ in range(20):
            self.emotional_memory.process_emotion('joy', 0.7, "Stable positive emotion")
            
        # Get emergence patterns
        patterns = self.emotional_memory.detect_emotional_patterns()
        
        # Verify pattern stability
        stability_scores = []
        for _ in range(5):
            new_patterns = self.emotional_memory.detect_emotional_patterns()
            stability_scores.append(len(new_patterns))
            
        # Verify consistent pattern detection
        self.assertTrue(all(score == stability_scores[0] for score in stability_scores))
        
if __name__ == '__main__':
    unittest.main() 