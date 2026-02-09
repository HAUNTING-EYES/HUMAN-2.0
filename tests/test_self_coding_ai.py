import unittest
import os
import tempfile
from pathlib import Path
from src.components.self_coding_ai import SelfCodingAI

class TestSelfCodingAI(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.ai = SelfCodingAI(self.test_dir)
        
        # Create test code file
        self.test_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
        self.test_file = Path(self.test_dir) / "test_code.py"
        with open(self.test_file, 'w') as f:
            f.write(self.test_code)
            
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test proper initialization of SelfCodingAI."""
        self.assertIsNotNone(self.ai)
        self.assertIsNotNone(self.ai.code_analyzer)
        self.assertIsNotNone(self.ai.self_improvement)
        self.assertIsNotNone(self.ai.continuous_learning)
        self.assertIsNotNone(self.ai.model)
        self.assertIsNotNone(self.ai.tokenizer)
        
    def test_analyze_code(self):
        """Test code analysis functionality."""
        result = self.ai.analyze_code(str(self.test_file))
        
        self.assertIsInstance(result, dict)
        self.assertIn('static_analysis', result)
        self.assertIn('dynamic_analysis', result)
        self.assertIn('suggestions', result)
        self.assertIn('timestamp', result)
        
    def test_improve_code(self):
        """Test code improvement functionality."""
        # First analyze the code
        analysis = self.ai.analyze_code(str(self.test_file))
        suggestions = analysis['suggestions']
        
        # Apply improvements
        result = self.ai.improve_code(str(self.test_file), suggestions)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('improvement_record', result)
            self.assertTrue(self.test_file.with_suffix('.py.backup').exists())
            
    def test_improvement_history(self):
        """Test improvement history tracking."""
        # Make some improvements
        analysis = self.ai.analyze_code(str(self.test_file))
        suggestions = analysis['suggestions']
        self.ai.improve_code(str(self.test_file), suggestions)
        
        history = self.ai.get_improvement_history()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        metrics = self.ai.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        
    def test_code_validation(self):
        """Test code validation functionality."""
        # Create some test code
        test_code = """
def test_function():
    return 42
"""
        validation_result = self.ai._validate_improvements(test_code)
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn('success', validation_result)
        self.assertIn('static_validation', validation_result)
        self.assertIn('dynamic_validation', validation_result)
        self.assertIn('has_regressions', validation_result)
        
    def test_improvement_prompt(self):
        """Test improvement prompt generation."""
        suggestion = {
            'description': 'Add type hints to function parameters'
        }
        prompt = self.ai._create_improvement_prompt(self.test_code, suggestion)
        
        self.assertIsInstance(prompt, str)
        self.assertIn(suggestion['description'], prompt)
        self.assertIn(self.test_code, prompt)
        
    def test_code_generation(self):
        """Test code generation functionality."""
        prompt = "Write a function that adds two numbers:"
        generated_code = self.ai._generate_code_improvement(prompt)
        
        self.assertIsInstance(generated_code, str)
        self.assertGreater(len(generated_code), 0)
        
if __name__ == '__main__':
    unittest.main() 