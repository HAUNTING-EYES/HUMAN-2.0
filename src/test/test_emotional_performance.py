import unittest
import time
import random
import numpy as np
from datetime import datetime
from src.components.emotional_integration import EmotionalIntegration

class TestEmotionalPerformance(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.integration = EmotionalIntegration(
            memory_size=10000,  # Large memory for stress testing
            learning_rate=0.01,
            influence_threshold=0.3,
            balance_threshold=0.5,
            adaptation_rate=0.1
        )
        
        # Define test parameters
        self.num_interactions = 1000  # Number of interactions for stress test
        self.max_processing_time = 0.1  # Maximum time per interaction (100ms)
        self.max_memory_usage = 100 * 1024 * 1024  # 100MB maximum memory usage
        
    def generate_random_state(self) -> dict:
        """Generate random emotional state."""
        return {
            'valence': random.uniform(-1.0, 1.0),
            'arousal': random.uniform(0.0, 1.0),
            'dominance': random.uniform(0.0, 1.0),
            'joy': random.uniform(0.0, 1.0),
            'sadness': random.uniform(0.0, 1.0),
            'anger': random.uniform(0.0, 1.0),
            'fear': random.uniform(0.0, 1.0),
            'surprise': random.uniform(0.0, 1.0)
        }
        
    def generate_random_interaction(self) -> dict:
        """Generate random interaction data."""
        return {
            'type': 'test',
            'timestamp': datetime.now().isoformat(),
            'emotional_state': self.generate_random_state(),
            'intensity': random.uniform(0.0, 1.0),
            'context': {
                'valence': random.uniform(-1.0, 1.0),
                'arousal': random.uniform(0.0, 1.0),
                'environment': random.choice(['positive', 'negative', 'neutral']),
                'social_context': random.choice(['individual', 'group', 'crowd'])
            }
        }
        
    def test_interaction_processing_speed(self):
        """Test processing speed of emotional interactions."""
        processing_times = []
        
        for _ in range(self.num_interactions):
            interaction = self.generate_random_interaction()
            
            # Measure processing time
            start_time = time.time()
            result = self.integration.process_interaction(interaction)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Verify successful processing
            self.assertTrue(result['success'])
            
        # Calculate statistics
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        p95_time = np.percentile(processing_times, 95)
        
        # Assert performance requirements
        self.assertLess(avg_time, self.max_processing_time / 2)  # Average should be under 50ms
        self.assertLess(max_time, self.max_processing_time)  # Max should be under 100ms
        self.assertLess(p95_time, self.max_processing_time * 0.8)  # 95th percentile under 80ms
        
    def test_memory_usage_under_load(self):
        """Test memory usage under heavy load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large number of interactions
        for _ in range(self.num_interactions):
            interaction = self.generate_random_interaction()
            self.integration.process_interaction(interaction)
            
            # Check memory usage
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Assert memory usage within limits
            self.assertLess(memory_increase, self.max_memory_usage)
            
    def test_pattern_detection_performance(self):
        """Test performance of pattern detection."""
        detection_times = []
        
        for _ in range(self.num_interactions):
            state1 = self.generate_random_state()
            state2 = self.generate_random_state()
            
            # Measure pattern detection time
            start_time = time.time()
            patterns = self.integration._detect_emergence_patterns(
                state1, state2, {'type': 'test'}
            )
            end_time = time.time()
            
            detection_time = end_time - start_time
            detection_times.append(detection_time)
            
        # Calculate statistics
        avg_time = np.mean(detection_times)
        max_time = np.max(detection_times)
        
        # Assert performance requirements
        self.assertLess(avg_time, 0.01)  # Average under 10ms
        self.assertLess(max_time, 0.05)  # Max under 50ms
        
    def test_emergence_effect_performance(self):
        """Test performance of emergence effect application."""
        effect_times = []
        
        for _ in range(self.num_interactions):
            state = self.generate_random_state()
            patterns = [
                {'type': 'resonance', 'strength': random.random()},
                {'type': 'contagion', 'strength': random.random()},
                {'type': 'regulation', 'strength': random.random()},
                {'type': 'adaptation', 'strength': random.random()}
            ]
            
            # Measure effect application time
            start_time = time.time()
            emerged_state = self.integration._apply_emergence_effects(
                state, patterns
            )
            end_time = time.time()
            
            effect_time = end_time - start_time
            effect_times.append(effect_time)
            
        # Calculate statistics
        avg_time = np.mean(effect_times)
        max_time = np.max(effect_times)
        
        # Assert performance requirements
        self.assertLess(avg_time, 0.01)  # Average under 10ms
        self.assertLess(max_time, 0.05)  # Max under 50ms
        
    def test_concurrent_interaction_processing(self):
        """Test processing multiple interactions concurrently."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def process_interaction():
            interaction = self.generate_random_interaction()
            result = self.integration.process_interaction(interaction)
            return result['success']
            
        # Process interactions concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda _: process_interaction(),
                range(self.num_interactions)
            ))
            
        # Verify all interactions processed successfully
        self.assertTrue(all(results))
        
        # Verify thread safety
        self.assertEqual(
            len(self.integration.interaction_history),
            self.num_interactions
        )
        
    def test_system_stability(self):
        """Test system stability under continuous load."""
        start_time = time.time()
        error_count = 0
        
        # Process interactions continuously for 60 seconds
        while time.time() - start_time < 60:
            try:
                interaction = self.generate_random_interaction()
                result = self.integration.process_interaction(interaction)
                
                if not result['success']:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                
        # Assert system stability
        self.assertLess(error_count, 10)  # Less than 10 errors in 60 seconds
        
    def test_memory_cleanup(self):
        """Test memory cleanup under load."""
        import gc
        
        # Process large number of interactions
        for _ in range(self.num_interactions):
            interaction = self.generate_random_interaction()
            self.integration.process_interaction(interaction)
            
        # Force garbage collection
        gc.collect()
        
        # Verify memory cleanup
        self.assertLessEqual(
            len(self.integration.interaction_history),
            self.integration.memory.memory_buffer_size
        )
        
if __name__ == '__main__':
    unittest.main() 