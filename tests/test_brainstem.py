import unittest
import os
import shutil
import tempfile
from src.components.brainstem import Brainstem

class TestBrainstem(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.brainstem = Brainstem(base_dir=self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        try:
            # Close ChromaDB connections
            if hasattr(self.brainstem, 'web_learning'):
                if hasattr(self.brainstem.web_learning, 'vectordb'):
                    self.brainstem.web_learning.vectordb._client.close()
                    
            # Close any other database connections
            if hasattr(self.brainstem, 'continuous_learning'):
                if hasattr(self.brainstem.continuous_learning, 'db'):
                    self.brainstem.continuous_learning.db.close()
                    
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")
            
        finally:
            # Remove test directory
            try:
                if os.path.exists(self.test_dir):
                    shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not remove test directory: {str(e)}")
    
    def test_initialization(self):
        """Test basic initialization of Brainstem."""
        self.assertIsNotNone(self.brainstem)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertIsNotNone(self.brainstem.qinn)
        self.assertIsNotNone(self.brainstem.web_learning)
        self.assertIsNotNone(self.brainstem.code_analyzer)
        
    def test_process_task(self):
        """Test task processing."""
        task = {
            "type": "test",
            "input": "Test input",
            "context": {}
        }
        result = self.brainstem.process_task(task)
        self.assertIsInstance(result, dict)
        self.assertTrue("success" in result)
        self.assertTrue("task_id" in result)
        
    def test_error_handling(self):
        """Test error handling for invalid input."""
        invalid_task = {
            "type": "invalid",
            "input": None
        }
        result = self.brainstem.process_task(invalid_task)
        self.assertFalse(result["success"])
        self.assertTrue("error" in result)
        
    def test_learning_task(self):
        """Test handling of learning tasks."""
        task = {
            'type': 'learning',
            'source': 'test_source',
            'content': 'test content'
        }
        result = self.brainstem.process_task(task)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('message', result)
        
    def test_analysis_task(self):
        """Test handling of analysis tasks."""
        task = {
            'type': 'analysis',
            'code': 'def test(): pass',
            'language': 'python'
        }
        result = self.brainstem.process_task(task)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('analysis', result)
        
    def test_improvement_task(self):
        """Test handling of improvement tasks."""
        task = {
            'type': 'improvement',
            'code': 'def test(): pass',
            'metrics': {'complexity': 0.5}
        }
        result = self.brainstem.process_task(task)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('improvements', result)
        
    def test_state_management(self):
        """Test state management functionality."""
        # Create a test task
        test_task = {
            'type': 'learning',
            'source': 'test',
            'content': 'test content'
        }
        
        # Process the task to generate state
        self.brainstem.process_task(test_task)
        
        # Test saving state
        state_file = os.path.join(self.test_dir, 'test_state.json')
        self.brainstem.save_state(state_file)
        self.assertTrue(os.path.exists(state_file))
        
        # Create a new brainstem instance
        new_brainstem = Brainstem(base_dir=self.test_dir)
        
        # Test loading state
        new_brainstem.load_state(state_file)
        self.assertIsNotNone(new_brainstem.current_task)
        self.assertIsInstance(new_brainstem.task_history, list)
        self.assertEqual(len(new_brainstem.task_history), 1)
        self.assertEqual(new_brainstem.task_history[0], test_task)
        
if __name__ == '__main__':
    unittest.main() 