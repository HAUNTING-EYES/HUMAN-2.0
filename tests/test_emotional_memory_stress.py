import unittest
import time
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor
from src.components.emotional_memory import EmotionalMemory
from src.components.emotional_learning import EmotionalLearning

class TestEmotionalMemoryStress(unittest.TestCase):
    def setUp(self):
        """Initialize systems before each test."""
        self.em = EmotionalMemory()
        self.el = EmotionalLearning()
        
    def test_concurrent_processing(self):
        """Test system performance under concurrent emotional processing."""
        num_threads = 4
        num_interactions = 100
        
        def process_batch(batch_id):
            results = []
            for i in range(num_interactions):
                # Generate random emotional input
                valence = random.uniform(-1.0, 1.0)
                text = f"Test interaction {batch_id}_{i} with valence {valence}"
                result = self.em.process_interaction(text)
                results.append(result)
            return results
            
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_batch, i) for i in range(num_threads)]
            all_results = [future.result() for future in futures]
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance
        self.assertLess(total_time, 30.0)  # Should process 400 interactions in under 30 seconds
        self.assertEqual(len(all_results), num_threads)
        self.assertEqual(sum(len(batch) for batch in all_results), num_threads * num_interactions)
        
    def test_memory_stress(self):
        """Test memory management under heavy load."""
        # Generate large number of interactions
        num_interactions = 10000
        start_time = time.time()
        
        for i in range(num_interactions):
            text = f"Memory stress test interaction {i}"
            self.em.process_interaction(text)
            
            # Verify memory constraints
            self.assertLessEqual(
                len(self.em.short_term_memory),
                self.em.memory_buffer_size
            )
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance
        self.assertLess(total_time, 300.0)  # Should process 10000 interactions in under 5 minutes
        
    def test_learning_stress(self):
        """Test emotional learning system under stress."""
        num_experiences = 1000
        emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
        
        start_time = time.time()
        
        for i in range(num_experiences):
            # Generate random emotional experience
            emotion = random.choice(emotions)
            intensity = random.uniform(0.1, 1.0)
            context = {
                'interaction_type': 'stress_test',
                'sequence': i,
                'environment': 'test'
            }
            
            # Process emotion and learn
            emotional_state = {
                'valence': random.uniform(-1.0, 1.0),
                'arousal': random.uniform(0.0, 1.0),
                'dominance': random.uniform(0.0, 1.0)
            }
            
            response = self.el.adapt_emotional_response(emotional_state, context)
            
            # Verify response structure
            self.assertIn('text', response)
            self.assertIn('style', response)
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance
        self.assertLess(total_time, 60.0)  # Should process 1000 learning experiences in under 1 minute
        
    def test_pattern_detection_stress(self):
        """Test pattern detection under heavy load."""
        num_patterns = 500
        pattern_length = 10
        
        start_time = time.time()
        
        # Generate repeated emotional patterns
        for i in range(num_patterns):
            # Create a repeating pattern
            for j in range(pattern_length):
                text = f"Pattern {i} step {j}"
                self.em.process_interaction(text)
            
            # Detect patterns
            patterns = self.em.detect_emotional_patterns()
            self.assertIsNotNone(patterns)
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance
        self.assertLess(total_time, 120.0)  # Should process 5000 interactions in under 2 minutes
        
    def test_emotional_adaptation_stress(self):
        """Test emotional adaptation under rapid changes."""
        num_adaptations = 1000
        start_time = time.time()
        
        previous_state = None
        for i in range(num_adaptations):
            # Generate random emotional state
            emotional_state = {
                'valence': random.uniform(-1.0, 1.0),
                'arousal': random.uniform(0.0, 1.0),
                'dominance': random.uniform(0.0, 1.0)
            }
            
            # Process adaptation
            context = {'interaction_type': 'adaptation_test', 'sequence': i}
            response = self.el.adapt_emotional_response(emotional_state, context)
            
            # Verify adaptation
            if previous_state:
                self.assertNotEqual(
                    response['style'],
                    previous_state['style']
                )
            
            previous_state = response
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance
        self.assertLess(total_time, 60.0)  # Should process 1000 adaptations in under 1 minute
        
    def test_system_resilience(self):
        """Test system resilience under error conditions."""
        num_tests = 1000
        error_count = 0
        
        for i in range(num_tests):
            try:
                # Generate potentially problematic input
                if i % 3 == 0:
                    text = ""  # Empty input
                elif i % 3 == 1:
                    text = "a" * 10000  # Very long input
                else:
                    text = None  # Invalid input
                    
                result = self.em.process_interaction(text)
                
                # System should handle all cases without crashing
                self.assertIsNotNone(result)
                
            except Exception as e:
                error_count += 1
                
        # System should handle most error cases
        self.assertLess(error_count / num_tests, 0.1)  # Less than 10% error rate
        
if __name__ == '__main__':
    unittest.main() 