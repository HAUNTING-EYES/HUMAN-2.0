import unittest
import torch
from src.components.logical_reasoning import LogicalReasoning

class TestLogicalReasoning(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.logical_reasoning = LogicalReasoning()
        
    def test_reasoning_with_context(self):
        """Test basic reasoning with context."""
        context = {
            "task": "analyze code",
            "parameters": {"language": "python"}
        }
        query = "What is the best approach?"
        
        result = self.logical_reasoning.reason(context, query)
        
        self.assertIsNotNone(result)
        self.assertIn("result", result)
        self.assertIn("confidence", result)
        self.assertIn("reasoning_chain", result)
        
    def test_symbolic_rule_application(self):
        """Test adding and applying symbolic rules."""
        rule = {
            "id": "test_rule",
            "description": "Test rule",
            "conditions": ["python"],
            "conclusion": "Use Python best practices"
        }
        
        self.logical_reasoning.add_symbolic_rule(rule)
        
        context = {
            "task": "analyze code",
            "parameters": {"language": "python"}
        }
        query = "What approach to use?"
        
        result = self.logical_reasoning.reason(context, query)
        
        self.assertIn("test_rule", result["result"]["applied_rules"])
        
    def test_causal_reasoning(self):
        """Test causal reasoning capabilities."""
        # Add causal relationship
        self.logical_reasoning.update_causal_graph(
            cause="complex_code",
            effect="maintenance_difficulty",
            probability=0.8
        )
        
        context = {
            "task": "analyze code",
            "parameters": {"complexity": "high"}
        }
        query = "What are the implications?"
        
        result = self.logical_reasoning.reason(context, query)
        
        self.assertIsNotNone(result)
        self.assertGreater(result["confidence"], 0.0)
        
    def test_uncertainty_handling(self):
        """Test uncertainty handling in reasoning."""
        context = {
            "task": "analyze code",
            "parameters": {"language": "unknown"}
        }
        query = "What approach to use?"
        
        result = self.logical_reasoning.reason(context, query)
        
        self.assertLess(result["confidence"], 0.7)
        self.assertIn("uncertain", result["result"]["status"].lower())
        
if __name__ == '__main__':
    unittest.main() 