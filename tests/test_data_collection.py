import unittest
import os
import json
import shutil
from src.data_collection.collect_data import collect_data
from src.components.web_learning import WebLearningSystem

class TestDataCollection(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        self.system = WebLearningSystem(base_dir=self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        try:
            # Close any open ChromaDB connections
            if hasattr(self, 'collector') and hasattr(self.collector, 'vectordb'):
                self.collector.vectordb.close()
            
            # Remove test directory
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception as e:
            self.logger.error(f"Error in tearDown: {str(e)}")
            
    def test_github_repository_collection(self):
        """Test collection from GitHub repositories."""
        test_repos = [
            'https://github.com/test/repo1',
            'https://github.com/test/repo2'
        ]
        
        # Mock GitHub token
        os.environ['GITHUB_TOKEN'] = 'test_token'
        
        # Collect data
        collect_data(github_repos=test_repos, base_dir=self.test_dir)
        
        # Verify cache files were created
        cache_dir = os.path.join(self.test_dir, 'cache')
        self.assertTrue(os.path.exists(cache_dir))
        cache_files = os.listdir(cache_dir)
        self.assertGreater(len(cache_files), 0)
        
    def test_documentation_collection(self):
        """Test collection from documentation sources."""
        test_docs = [
            'https://docs.test.com/doc1',
            'https://docs.test.com/doc2'
        ]
        
        # Collect data
        collect_data(doc_urls=test_docs, base_dir=self.test_dir)
        
        # Verify documentation was collected
        doc_dir = os.path.join(self.test_dir, 'documentation')
        self.assertTrue(os.path.exists(doc_dir))
        doc_files = os.listdir(doc_dir)
        self.assertGreater(len(doc_files), 0)
        
    def test_code_pattern_extraction(self):
        """Test code pattern extraction from collected data."""
        test_code = """
        def example_function():
            # This is a test function
            return True
        """
        
        # Save test code
        pattern_dir = os.path.join(self.test_dir, 'code_patterns')
        os.makedirs(pattern_dir, exist_ok=True)
        
        with open(os.path.join(pattern_dir, 'test.py'), 'w') as f:
            f.write(test_code)
            
        # Verify pattern extraction
        self.system._extract_code_patterns(pattern_dir)
        self.assertTrue(os.path.exists(os.path.join(pattern_dir, 'patterns.json')))
        
    def test_example_generation(self):
        """Test example generation from collected data."""
        test_data = {
            'code': 'def test(): pass',
            'description': 'Test function',
            'tags': ['test', 'example']
        }
        
        # Save test data
        example_dir = os.path.join(self.test_dir, 'examples')
        os.makedirs(example_dir, exist_ok=True)
        
        with open(os.path.join(example_dir, 'test.json'), 'w') as f:
            json.dump(test_data, f)
            
        # Verify example generation
        self.system._generate_examples(example_dir)
        self.assertTrue(os.path.exists(os.path.join(example_dir, 'generated_examples.json')))
        
    def test_error_handling(self):
        """Test error handling during data collection."""
        # Test with invalid GitHub token
        os.environ['GITHUB_TOKEN'] = 'invalid_token'
        
        # Should handle error gracefully
        collect_data(github_repos=['https://github.com/test/repo'], base_dir=self.test_dir)
        
        # Test with invalid documentation URL
        collect_data(doc_urls=['https://invalid.url'], base_dir=self.test_dir)
        
        # Verify system remains in a valid state
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.exists(self.system.cache_dir))

if __name__ == '__main__':
    unittest.main() 