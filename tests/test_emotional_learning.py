#!/usr/bin/env python3
"""Tests for the emotional learning system."""

import unittest
import numpy as np
import torch
import os
import tempfile
import shutil
from pathlib import Path
from src.components.emotional_learning import EmotionalLearningSystem

class TestEmotionalLearning(unittest.TestCase):
    """Test cases for the EmotionalLearningSystem class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.learning_system = EmotionalLearningSystem(
            state_size=8,
            action_size=8,
            learning_rate=0.001,
            base_dir=Path(self.temp_dir)
        )
        
    def tearDown(self):
        """Clean up after tests."""
        try:
            # Use shutil.rmtree for more reliable cleanup on Windows
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {e}")
        
    def test_initialization(self):
        """Test initialization of the learning system."""
        self.assertIsNotNone(self.learning_system.model)
        self.assertEqual(len(self.learning_system.memory), 0)
        self.assertEqual(self.learning_system.epsilon, 1.0)
        self.assertEqual(self.learning_system.current_strategy, 'balanced')
        
    def test_remember(self):
        """Test storing experiences in memory."""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        action = 2
        reward = 0.5
        next_state = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        done = False
        
        self.learning_system.remember(state, action, reward, next_state, done)
        self.assertEqual(len(self.learning_system.memory), 1)
        
    def test_act(self):
        """Test action selection."""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        # Test exploration (epsilon = 1.0)
        action = self.learning_system.act(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.learning_system.action_size)
        
        # Test exploitation (epsilon = 0.0)
        self.learning_system.epsilon = 0.0
        action = self.learning_system.act(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.learning_system.action_size)
        
    def test_replay(self):
        """Test experience replay training."""
        # Add some experiences to memory
        for _ in range(10):
            state = np.random.rand(8)
            action = np.random.randint(0, 8)
            reward = np.random.rand()
            next_state = np.random.rand(8)
            done = np.random.choice([True, False])
            self.learning_system.remember(state, action, reward, next_state, done)
            
        # Test replay with batch size 5
        self.learning_system.replay(5)
        
        # Epsilon should have decreased
        self.assertLess(self.learning_system.epsilon, 1.0)
        
    def test_learn_from_interaction(self):
        """Test learning from an interaction."""
        interaction_data = {
            'emotional_state': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'response_index': 2,
            'next_emotional_state': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'response_appropriateness': 0.8,
            'emotional_stability': 0.7,
            'empathy_effectiveness': 0.9,
            'emotional_intensity': 0.6,
            'personality_consistency': 0.8
        }
        
        reward = self.learning_system.learn_from_interaction(interaction_data)
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)
        
    def test_emotional_strategy_generation(self):
        """Test generation of emotional strategies."""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        strategy = self.learning_system.get_emotional_strategy(state)
        
        self.assertIsInstance(strategy, dict)
        self.assertIn('joy', strategy)
        self.assertIn('sadness', strategy)
        self.assertIn('anger', strategy)
        self.assertIn('fear', strategy)
        self.assertIn('surprise', strategy)
        self.assertIn('disgust', strategy)
        self.assertIn('neutral', strategy)
        self.assertIn('complex', strategy)
        
        # Check that probabilities sum to approximately 1
        self.assertAlmostEqual(sum(strategy.values()), 1.0, places=1)
        
    def test_model_save_load(self):
        """Test saving and loading the model."""
        # Save model
        save_path = os.path.join(self.temp_dir, 'model.pt')
        self.learning_system.save_model(save_path)
        
        # Create a new instance
        new_system = EmotionalLearningSystem(
            state_size=8,
            action_size=8,
            learning_rate=0.001,
            base_dir=Path(self.temp_dir)
        )
        
        # Load model
        new_system.load_model(save_path)
        
        # Check that models have same parameters
        for p1, p2 in zip(self.learning_system.model.parameters(), new_system.model.parameters()):
            self.assertTrue(torch.equal(p1, p2))
            
    def test_pattern_detection(self):
        """Test detection of emotional patterns."""
        # Create emotional history
        emotional_history = [
            {'joy': 0.1, 'sadness': 0.2, 'anger': 0.3, 'fear': 0.4, 'surprise': 0.5, 'disgust': 0.6},
            {'joy': 0.2, 'sadness': 0.3, 'anger': 0.4, 'fear': 0.5, 'surprise': 0.6, 'disgust': 0.7},
            {'joy': 0.3, 'sadness': 0.4, 'anger': 0.5, 'fear': 0.6, 'surprise': 0.7, 'disgust': 0.8},
            {'joy': 0.4, 'sadness': 0.5, 'anger': 0.6, 'fear': 0.7, 'surprise': 0.8, 'disgust': 0.9}
        ]
        
        patterns = self.learning_system.detect_patterns(emotional_history)
        
        self.assertIsInstance(patterns, list)
        
    def test_strategy_adaptation(self):
        """Test adaptation of emotional strategies."""
        emotional_state = {
            'joy': 0.1, 'sadness': 0.2, 'anger': 0.3, 'fear': 0.4,
            'surprise': 0.5, 'disgust': 0.6, 'neutral': 0.7, 'complex': 0.8
        }
        
        interaction_history = [
            {'sentiment': 0.8, 'outcome': 'positive'},
            {'sentiment': 0.7, 'outcome': 'positive'},
            {'sentiment': 0.6, 'outcome': 'positive'}
        ]
        
        strategy = self.learning_system.adapt_strategy(emotional_state, interaction_history)
        
        self.assertIn(strategy, ['defensive', 'aggressive', 'balanced'])
        
    def test_emotional_response_generation(self):
        """Test generation of emotional responses."""
        emotional_state = {
            'joy': 0.1, 'sadness': 0.2, 'anger': 0.3, 'fear': 0.4,
            'surprise': 0.5, 'disgust': 0.6, 'neutral': 0.7, 'complex': 0.8
        }
        
        context = {
            'interaction_type': 'positive',
            'user_emotion': 'joy',
            'intensity': 0.7
        }
        
        response = self.learning_system.adapt_emotional_response(emotional_state, context)
        
        self.assertIsInstance(response, dict)
        self.assertIn('text', response)
        self.assertIn('style', response)
        self.assertIsInstance(response['text'], str)
        self.assertIsInstance(response['style'], dict)
        
    def test_learning_state_persistence(self):
        """Test persistence of learning state."""
        # Save learning state
        save_path = os.path.join(self.temp_dir, 'learning_state.pt')
        self.learning_system.save_learning_state(save_path)
        
        # Create a new instance
        new_system = EmotionalLearningSystem(
            state_size=8,
            action_size=8,
            learning_rate=0.001,
            base_dir=Path(self.temp_dir)
        )
        
        # Load learning state
        new_system.load_learning_state(save_path)
        
        # Check that epsilon is the same
        self.assertEqual(self.learning_system.epsilon, new_system.epsilon)
        
    def test_learning_from_experience(self):
        """Test learning from emotional experiences."""
        emotion = 'joy'
        intensity = 0.7
        context = {'situation': 'positive_feedback', 'source': 'user'}
        outcome = {'success': True, 'satisfaction': 0.8, 'adaptation': 0.7}
        
        result = self.learning_system.learn_from_experience(emotion, intensity, context, outcome)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('patterns_updated', result)
        self.assertIn('weights_updated', result)
        
    def test_learning_status(self):
        """Test retrieval of learning status."""
        status = self.learning_system.get_learning_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('patterns_count', status)
        self.assertIn('history_size', status)
        self.assertIn('learning_rate', status)
        self.assertIn('discount_factor', status)
        self.assertIn('pattern_weights', status)
        self.assertIn('timestamp', status)
        
    def test_learning_reset(self):
        """Test resetting the learning state."""
        # Add some data
        self.learning_system.emotional_patterns = [{'id': 'test_pattern'}]
        self.learning_system.pattern_weights = {'test_pattern': 0.5}
        self.learning_system.learning_history = [{'test': 'data'}]
        self.learning_system.q_table = {'test': 'data'}
        self.learning_system.experience_buffer = [{'test': 'data'}]
        
        # Reset learning state
        result = self.learning_system.reset_learning()
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertTrue(result['success'])
        self.assertEqual(len(self.learning_system.emotional_patterns), 0)
        self.assertEqual(len(self.learning_system.pattern_weights), 0)
        self.assertEqual(len(self.learning_system.learning_history), 0)
        self.assertEqual(len(self.learning_system.q_table), 0)
        self.assertEqual(len(self.learning_system.experience_buffer), 0)

if __name__ == '__main__':
    unittest.main() 