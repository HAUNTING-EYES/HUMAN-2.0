import unittest
import numpy as np
from src.core.pattern_recognition import PatternRecognition

class TestPatternRecognition(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.pattern_recognition = PatternRecognition()
        self.test_patterns = [
            {'type': 'code', 'content': 'def test_function():\n    return True'},
            {'type': 'emotion', 'content': {'joy': 0.8, 'sadness': 0.2}},
            {'type': 'behavior', 'content': ['learn', 'adapt', 'improve']}
        ]
        
    def test_initialization(self):
        """Test proper initialization of pattern recognition system."""
        self.assertIsNotNone(self.pattern_recognition.pattern_db)
        self.assertIsNotNone(self.pattern_recognition.recognition_model)
        self.assertEqual(len(self.pattern_recognition.active_patterns), 0)
        
    def test_pattern_detection(self):
        """Test pattern detection capabilities."""
        for pattern in self.test_patterns:
            result = self.pattern_recognition.detect_pattern(pattern['content'])
            self.assertIsInstance(result, dict)
            self.assertIn('confidence', result)
            self.assertIn('pattern_type', result)
            self.assertTrue(0 <= result['confidence'] <= 1)
            
    def test_pattern_learning(self):
        """Test pattern learning capabilities."""
        initial_patterns = len(self.pattern_recognition.pattern_db)
        
        # Learn new patterns
        for pattern in self.test_patterns:
            self.pattern_recognition.learn_pattern(pattern['content'], pattern['type'])
            
        # Verify patterns were learned
        self.assertGreater(len(self.pattern_recognition.pattern_db), initial_patterns)
        
    def test_pattern_evolution(self):
        """Test pattern evolution over time."""
        # Add initial patterns
        for pattern in self.test_patterns:
            self.pattern_recognition.learn_pattern(pattern['content'], pattern['type'])
            
        # Evolve patterns
        evolved_patterns = self.pattern_recognition.evolve_patterns()
        
        self.assertIsInstance(evolved_patterns, list)
        self.assertGreater(len(evolved_patterns), 0)
        
    def test_pattern_matching(self):
        """Test pattern matching functionality."""
        # Add patterns to match against
        for pattern in self.test_patterns:
            self.pattern_recognition.learn_pattern(pattern['content'], pattern['type'])
            
        # Test matching
        test_input = {'type': 'code', 'content': 'def similar_function():\n    return False'}
        matches = self.pattern_recognition.find_matches(test_input)
        
        self.assertIsInstance(matches, list)
        self.assertTrue(all(0 <= match['similarity'] <= 1 for match in matches))
        
    def test_pattern_validation(self):
        """Test pattern validation checks."""
        # Valid pattern
        valid_pattern = {'type': 'code', 'content': 'def valid_function():\n    pass'}
        self.assertTrue(self.pattern_recognition.validate_pattern(valid_pattern))
        
        # Invalid patterns
        invalid_patterns = [
            None,
            {},
            {'type': 'unknown'},
            {'content': 'no type specified'},
            {'type': 'code', 'content': ''}
        ]
        
        for invalid_pattern in invalid_patterns:
            self.assertFalse(self.pattern_recognition.validate_pattern(invalid_pattern))
            
    def test_pattern_persistence(self):
        """Test pattern storage and retrieval."""
        # Store patterns
        for pattern in self.test_patterns:
            self.pattern_recognition.store_pattern(pattern)
            
        # Retrieve and verify patterns
        for pattern in self.test_patterns:
            retrieved = self.pattern_recognition.retrieve_pattern(pattern['type'])
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved['type'], pattern['type'])
            
    def test_pattern_analysis(self):
        """Test pattern analysis capabilities."""
        # Add patterns for analysis
        for pattern in self.test_patterns:
            self.pattern_recognition.learn_pattern(pattern['content'], pattern['type'])
            
        # Analyze patterns
        analysis = self.pattern_recognition.analyze_patterns()
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('pattern_count', analysis)
        self.assertIn('pattern_types', analysis)
        self.assertIn('pattern_distribution', analysis)
        
if __name__ == '__main__':
    unittest.main() 