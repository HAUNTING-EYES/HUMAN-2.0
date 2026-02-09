import unittest
import os
import json
import shutil
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from src.components.web_learning import WebLearningSystem

class TestWebLearningSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        self.system = WebLearningSystem(base_dir=self.test_dir)
        self.web_learning = WebLearningSystem()
        self.test_url = "https://example.com"
        self.test_content = """
        def example_function():
            print("Hello, World!")
            
        class ExampleClass:
            def __init__(self):
                self.value = 42
        """
        
    def tearDown(self):
        """Clean up test environment."""
        try:
            # Close ChromaDB connections
            if hasattr(self, 'system') and hasattr(self.system, 'vectordb'):
                if hasattr(self.system.vectordb, '_client'):
                    self.system.vectordb._client.close()
                    
            # Remove test directory
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
        except Exception as e:
            print(f"Error in tearDown: {str(e)}")
            
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.base_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.system.cache_dir))
        self.assertTrue(os.path.exists(self.system.code_patterns_dir))
        self.assertTrue(os.path.exists(self.system.documentation_dir))
        self.assertTrue(os.path.exists(self.system.examples_dir))
        
        # Check if base_dir is set correctly
        self.assertIsNotNone(self.web_learning.base_dir)
        self.assertTrue(isinstance(self.web_learning.base_dir, Path))
        
        # Check if visited_urls is initialized
        self.assertIsNotNone(self.web_learning.visited_urls)
        self.assertTrue(isinstance(self.web_learning.visited_urls, set))
        
        # Check if zero_shot_classifier is initialized
        self.assertIsNotNone(self.web_learning.zero_shot_classifier)
        
    def test_numpy_conversion(self):
        """Test numpy array conversion for JSON serialization."""
        test_data = {
            'array': np.array([1, 2, 3]),
            'nested': {
                'array': np.array([[1, 2], [3, 4]]),
                'list': [np.array([5, 6]), np.array([7, 8])]
            }
        }
        
        converted = self.system._convert_numpy_to_list(test_data)
        self.assertIsInstance(converted['array'], list)
        self.assertIsInstance(converted['nested']['array'], list)
        self.assertIsInstance(converted['nested']['list'][0], list)
        
    def test_cache_operations(self):
        """Test cache saving and loading."""
        test_data = {
            'test': 'data',
            'array': np.array([1, 2, 3])
        }
        
        # Test saving
        cache_file = self.system._save_to_cache('test_key', test_data)
        self.assertTrue(os.path.exists(cache_file))
        
        # Test loading
        loaded_data = self.system._get_from_cache('test_key')
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data['data']['test'], 'data')
        
    def test_cache_expiry(self):
        """Test cache expiration."""
        test_data = {'test': 'data'}
        
        # Save with old timestamp
        old_time = datetime.now() - timedelta(hours=25)
        cache_file = self.system._save_to_cache('test_key', test_data, timestamp=old_time)
        
        # Should return None due to expiration
        loaded_data = self.system._get_from_cache('test_key')
        self.assertIsNone(loaded_data)
        
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        start_time = datetime.now()
        
        # Make two requests
        self.system._rate_limit()
        self.system._rate_limit()
        
        end_time = datetime.now()
        time_diff = (end_time - start_time).total_seconds()
        
        # Should have waited at least min_request_interval
        self.assertGreaterEqual(time_diff, self.system.min_request_interval)
        
    def test_url_processing(self):
        """Test URL processing and duplicate prevention."""
        test_url = 'https://github.com/test/repo'
        
        # First visit
        self.system.learn_from_url(test_url)
        self.assertIn(test_url, self.system.visited_urls)
        
        # Second visit should be skipped
        self.system.learn_from_url(test_url)
        self.assertEqual(len(self.system.visited_urls), 1)
        
    def test_nlp_components(self):
        """Test NLP component initialization and functionality."""
        self.assertIsNotNone(self.system.sentiment_analyzer)
        self.assertIsNotNone(self.system.zero_shot_classifier)
        
        # Test sentiment analysis
        sentiment = self.system.sentiment_analyzer("This is great code!")[0]
        self.assertIn('label', sentiment)
        self.assertIn('score', sentiment)
        
        # Test zero-shot classification
        result = self.system.zero_shot_classifier(
            sequences="This function implements a binary search algorithm",
            candidate_labels=["algorithm", "data structure", "utility"]
        )
        self.assertIn('labels', result)
        self.assertIn('scores', result)

    def test_learn_from_url(self):
        """Test learning from URL."""
        # Mock URL content
        result = self.web_learning.learn_from_url(self.test_url)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('url', result)
        self.assertIn('processed_chunks', result)
        self.assertIn('classifications', result)
        
        # Check if URL was added to visited_urls
        self.assertIn(self.test_url, self.web_learning.visited_urls)
        
    def test_extract_code_patterns(self):
        """Test code pattern extraction."""
        patterns = self.web_learning.extract_code_patterns(self.test_content)
        
        # Check if patterns were extracted
        self.assertIsInstance(patterns, dict)
        self.assertIn('functions', patterns)
        self.assertIn('classes', patterns)
        self.assertIn('imports', patterns)
        
        # Check if example function was found
        self.assertTrue(any('example_function' in func for func in patterns['functions']))
        
        # Check if example class was found
        self.assertTrue(any('ExampleClass' in cls for cls in patterns['classes']))
        
    def test_duplicate_url(self):
        """Test handling of duplicate URLs."""
        # First visit
        self.web_learning.learn_from_url(self.test_url)
        
        # Second visit
        result = self.web_learning.learn_from_url(self.test_url)
        
        # Check if second visit was skipped
        self.assertFalse(result['success'])
        self.assertIn('already visited', result.get('error', '').lower())

if __name__ == '__main__':
    unittest.main() 