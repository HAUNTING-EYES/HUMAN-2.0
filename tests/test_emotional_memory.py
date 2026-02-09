import unittest
import torch
from pathlib import Path
import tempfile
import shutil
from src.components.emotional_memory import EmotionalMemory
from datetime import datetime
import numpy as np
import time

class TestEmotionalMemory(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.emotional_memory = EmotionalMemory(base_dir=self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test initialization of EmotionalMemory."""
        # Test with default base_dir
        em_default = EmotionalMemory()
        self.assertIsNotNone(em_default.base_dir)
        self.assertIsNotNone(em_default.memory_dir)
        
        # Test with custom base_dir
        em_custom = EmotionalMemory(base_dir=self.test_dir)
        self.assertEqual(em_custom.base_dir, Path(self.test_dir))
        self.assertEqual(em_custom.memory_dir, Path(self.test_dir) / "emotional_memory")
        
        # Test emotional state initialization
        self.assertIn('joy', em_custom.emotional_state)
        self.assertIn('sadness', em_custom.emotional_state)
        self.assertIn('anger', em_custom.emotional_state)
        self.assertIn('fear', em_custom.emotional_state)
        self.assertIn('surprise', em_custom.emotional_state)
        self.assertIn('valence', em_custom.emotional_state)
        self.assertIn('arousal', em_custom.emotional_state)
        self.assertIn('dominance', em_custom.emotional_state)
        self.assertIn('timestamp', em_custom.emotional_state)
        
    def test_process_interaction(self):
        """Test processing interactions."""
        # Test with empty text
        result = self.emotional_memory.process_interaction("")
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(result['error'], "Empty text input")
        
        # Test with valid text
        result = self.emotional_memory.process_interaction("I am happy to help you with your code.")
        self.assertTrue(result['success'])
        self.assertIn('emotional_state', result)
        self.assertIn('emotional_response', result)
        self.assertIn('memory_count', result)
        self.assertIn('current_state', result)
        
    def test_emotional_state_updates(self):
        """Test emotional state updates."""
        # Reset emotional state
        self.emotional_memory.emotional_state = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'valence': 0.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process positive interaction
        positive_result = self.emotional_memory.process_interaction(
            "Great work! The code looks excellent and shows amazing progress!"
        )
        positive_state = positive_result['emotional_state']
        
        # Verify positive state changes
        self.assertGreater(positive_state['joy'], 0.2)
        self.assertLess(positive_state['sadness'], 0.1)
        self.assertGreater(positive_state['valence'], 0)
        
        # Reset emotional state
        self.emotional_memory.emotional_state = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'valence': 0.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process negative interaction
        negative_result = self.emotional_memory.process_interaction(
            "This code has serious issues and needs significant improvement."
        )
        negative_state = negative_result['emotional_state']
        
        # Verify negative state changes
        self.assertLess(negative_state['joy'], 0.1)
        self.assertGreater(negative_state['sadness'], 0.2)
        self.assertLess(negative_state['valence'], 0)
        
    def test_memory_management(self):
        """Test memory management."""
        # Add multiple interactions
        for i in range(5):
            self.emotional_memory.process_interaction(f"Test interaction {i}")
            
        # Check memory count
        current_state = self.emotional_memory._get_current_state()
        self.assertLessEqual(len(self.emotional_memory.memory_buffer), 100)
        self.assertEqual(current_state['memory_count'], len(self.emotional_memory.memory_buffer))
        
        # Test memory persistence
        new_em = EmotionalMemory(base_dir=self.test_dir)
        self.assertEqual(len(new_em.memory_buffer), len(self.emotional_memory.memory_buffer))
        
    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        # Test positive sentiment
        positive_text = "This is excellent code, very well written!"
        positive_sentiment = self.emotional_memory._analyze_sentiment(positive_text)
        
        # Test negative sentiment
        negative_text = "This code is poorly written and needs improvement."
        negative_sentiment = self.emotional_memory._analyze_sentiment(negative_text)
        
        # Verify sentiment analysis
        self.assertGreater(positive_sentiment, 0)
        self.assertLess(negative_sentiment, 0)
        
    def test_error_handling(self):
        """Test error handling."""
        # Test with invalid sentiment analyzer
        self.emotional_memory.sentiment_analyzer = None
        result = self.emotional_memory.process_interaction("Test text")
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        
    def test_clear_memories(self):
        """Test clearing memories."""
        # Add some memories
        for i in range(3):
            self.emotional_memory.process_interaction(f"Test interaction {i}")
            
        # Clear memories
        self.emotional_memory.clear_memories()
        
        # Verify memories are cleared
        self.assertEqual(len(self.emotional_memory.memory_buffer), 0)
        
        # Verify persistence
        new_em = EmotionalMemory(base_dir=self.test_dir)
        self.assertEqual(len(new_em.memory_buffer), 0)
        
    def test_emotional_dimensions(self):
        """Test initialization and handling of emotional dimensions."""
        # Check all dimensions are present
        expected_dimensions = {
            'valence', 'arousal', 'dominance', 'novelty',
            'complexity', 'intensity', 'stability', 'coherence'
        }
        self.assertEqual(set(self.emotional_memory.emotion_dimensions.keys()),
                        expected_dimensions)
        
        # Check dimension constraints
        for dimension, config in self.emotional_memory.emotion_dimensions.items():
            self.assertLess(config['min'], config['max'])
            self.assertGreaterEqual(config['default'], config['min'])
            self.assertLessEqual(config['default'], config['max'])
            
    def test_emotional_diffusion(self):
        """Test emotion diffusion with new dimensions."""
        # Create test emotional states
        current_emotion = {
            'valence': 0.6,
            'arousal': 0.4,
            'dominance': 0.5,
            'novelty': 0.3,
            'complexity': 0.7,
            'intensity': 0.5,
            'stability': 0.8,
            'coherence': 0.6
        }
        
        context_emotion = {
            'valence': 0.8,
            'arousal': 0.6,
            'dominance': 0.7,
            'novelty': 0.5,
            'complexity': 0.9,
            'intensity': 0.7,
            'stability': 0.6,
            'coherence': 0.8
        }
        
        # Test diffusion
        diffused = self.emotional_memory.diffuse_emotion(
            current_emotion, context_emotion, time_step=0.5
        )
        
        # Verify diffusion results
        for dimension in current_emotion:
            # Check values are within bounds
            self.assertGreaterEqual(diffused[dimension],
                                  self.emotional_memory.emotion_dimensions[dimension]['min'])
            self.assertLessEqual(diffused[dimension],
                               self.emotional_memory.emotion_dimensions[dimension]['max'])
            
            # Check diffusion direction
            if current_emotion[dimension] < context_emotion[dimension]:
                self.assertGreater(diffused[dimension], current_emotion[dimension])
            elif current_emotion[dimension] > context_emotion[dimension]:
                self.assertLess(diffused[dimension], current_emotion[dimension])
                
    def test_emergence_patterns(self):
        """Test detection and application of emergence patterns."""
        # Create emotional state with clear patterns
        emotion_state = {
            'valence': 0.9,
            'arousal': 0.9,
            'dominance': 0.9,
            'novelty': 0.1,
            'complexity': 0.1,
            'intensity': 0.9,
            'stability': 0.1,
            'coherence': 0.1
        }
        
        # Detect patterns
        patterns = self.emotional_memory.detect_emergence_patterns(emotion_state)
        
        # Verify pattern detection
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Apply pattern effects
        modified_state = self.emotional_memory.apply_emergence_effects(
            emotion_state, patterns
        )
        
        # Verify modifications
        self.assertNotEqual(emotion_state, modified_state)
        for dimension in emotion_state:
            self.assertGreaterEqual(modified_state[dimension],
                                  self.emotional_memory.emotion_dimensions[dimension]['min'])
            self.assertLessEqual(modified_state[dimension],
                               self.emotional_memory.emotion_dimensions[dimension]['max'])
            
    def test_emotional_inertia(self):
        """Test emotional inertia behavior."""
        # Test with extreme values
        current_val = 0.9
        dimension = 'valence'
        
        # Get inertia factor
        inertia = self.emotional_memory._apply_emotional_inertia(current_val, dimension)
        
        # Verify inertia properties
        self.assertGreaterEqual(inertia, 0.3)
        self.assertLessEqual(inertia, 0.9)
        
        # Test with neutral value
        neutral_inertia = self.emotional_memory._apply_emotional_inertia(0.5, dimension)
        self.assertGreater(neutral_inertia, inertia)  # Should have more inertia at neutral
        
    def test_personality_initialization(self):
        """Test personality trait initialization."""
        # Get initial personality profile
        profile = self.emotional_memory.get_personality_profile()
        
        # Check if all traits are initialized
        self.assertIn('traits', profile)
        self.assertIn('dominant_trait', profile)
        self.assertIn('emotional_style', profile)
        self.assertIn('interaction_preferences', profile)
        
        # Verify trait values are within bounds
        for trait, value in profile['traits'].items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            
    def test_emotional_processing(self):
        """Test emotion processing with personality effects."""
        # Process a positive emotion
        joy_state = self.emotional_memory.process_emotion('joy', 0.8, "Feeling happy about success")
        
        # Verify state structure
        self.assertIn('emotion', joy_state)
        self.assertIn('intensity', joy_state)
        self.assertIn('personality_influences', joy_state)
        self.assertIn('context', joy_state)
        self.assertIn('timestamp', joy_state)
        
        # Verify intensity is modified by personality
        self.assertGreater(joy_state['intensity'], 0.0)
        self.assertLessEqual(joy_state['intensity'], 1.0)
        
        # Process a negative emotion
        sadness_state = self.emotional_memory.process_emotion('sadness', 0.6, "Feeling down about a loss")
        
        # Verify personality influences are present
        self.assertIn('extraversion', sadness_state['personality_influences'])
        self.assertIn('neuroticism', sadness_state['personality_influences'])
        
    def test_emotional_response_generation(self):
        """Test emotional response generation."""
        # Generate response to joy
        joy_response = self.emotional_memory.generate_emotional_response('joy', "Celebrating success")
        
        # Verify response structure
        self.assertIn('emotion', joy_response)
        self.assertIn('intensity', joy_response)
        self.assertIn('style', joy_response)
        self.assertIn('context', joy_response)
        self.assertIn('personality_influences', joy_response)
        
        # Verify response emotion is appropriate
        self.assertIn(joy_response['emotion'], ['joy', 'contentment', 'optimism'])
        
        # Generate response to sadness
        sadness_response = self.emotional_memory.generate_emotional_response('sadness', "Dealing with loss")
        
        # Verify response emotion is appropriate
        self.assertIn(sadness_response['emotion'], ['empathy', 'sadness', 'concern'])
        
    def test_personality_trait_updates(self):
        """Test personality trait updates from emotional experiences."""
        initial_profile = self.emotional_memory.get_personality_profile()
        initial_traits = initial_profile['traits'].copy()
        
        # Process multiple emotions
        self.emotional_memory.process_emotion('joy', 0.8, "Happy experience")
        self.emotional_memory.process_emotion('sadness', 0.6, "Sad experience")
        self.emotional_memory.process_emotion('anger', 0.7, "Angry experience")
        
        # Get updated profile
        updated_profile = self.emotional_memory.get_personality_profile()
        updated_traits = updated_profile['traits']
        
        # Verify traits have been updated
        for trait in initial_traits:
            self.assertNotEqual(initial_traits[trait], updated_traits[trait])
            
    def test_emotional_style_calculation(self):
        """Test emotional style calculation."""
        profile = self.emotional_memory.get_personality_profile()
        style = profile['emotional_style']
        
        # Verify emotional style components
        self.assertIn('emotional_intensity', style)
        self.assertIn('emotional_stability', style)
        self.assertIn('emotional_expressiveness', style)
        
        # Verify values are within bounds
        for component, value in style.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            
    def test_interaction_preferences(self):
        """Test interaction preferences calculation."""
        profile = self.emotional_memory.get_personality_profile()
        preferences = profile['interaction_preferences']
        
        # Verify preference components
        self.assertIn('social_engagement', preferences)
        self.assertIn('emotional_support', preferences)
        self.assertIn('adaptability', preferences)
        
        # Verify values are within bounds
        for preference, value in preferences.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            
    def test_emotional_distance(self):
        """Test emotional distance calculation."""
        emotion1 = {'valence': 0.8, 'arousal': 0.6, 'dominance': 0.7}
        emotion2 = {'valence': -0.6, 'arousal': -0.3, 'dominance': -0.4}
        
        distance = self.emotional_memory._calculate_emotional_distance(emotion1, emotion2)
        
        # Verify distance is calculated correctly
        self.assertGreater(distance, 0.0)
        self.assertLessEqual(distance, 1.0)
        
    def test_empathy_simulation(self):
        """Test empathy simulation."""
        # Test empathy for positive emotion
        joy_response = self.emotional_memory.simulate_empathy('joy', "I'm feeling really happy today!")
        self.assertIsInstance(joy_response, str)
        self.assertGreater(len(joy_response), 0)
        
        # Test empathy for negative emotion
        sadness_response = self.emotional_memory.simulate_empathy('sadness', "I'm feeling down today.")
        self.assertIsInstance(sadness_response, str)
        self.assertGreater(len(sadness_response), 0)
        
        # Test empathy level calculation
        empathy_level = self.emotional_memory._calculate_empathy_level()
        self.assertGreaterEqual(empathy_level, 0.0)
        self.assertLessEqual(empathy_level, 1.0)

if __name__ == '__main__':
    unittest.main() 