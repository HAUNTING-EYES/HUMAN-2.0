import unittest
import numpy as np
import gymnasium as gym
from src.components.self_coding_ai import CodeOptimizationEnv

class TestCodeOptimizationEnv(unittest.TestCase):
    def setUp(self):
        self.initial_code = "def test_function():\n    return 42"
        self.env = CodeOptimizationEnv(self.initial_code)
        
    def test_initialization(self):
        """Test environment initialization"""
        self.assertIsInstance(self.env, gym.Env)
        self.assertEqual(self.env.initial_code, self.initial_code)
        self.assertEqual(self.env.current_code, self.initial_code)
        self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.env.max_steps, 10)
        
    def test_reset(self):
        """Test environment reset"""
        # Modify environment state
        self.env.current_code = "modified code"
        self.env.steps = 5
        
        # Reset environment
        state, info = self.env.reset()
        
        # Check reset state
        self.assertEqual(self.env.current_code, self.initial_code)
        self.assertEqual(self.env.steps, 0)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (1000,))
        self.assertEqual(info, {})
        
    def test_step(self):
        """Test environment step"""
        # Take a step
        state, reward, terminated, truncated, info = self.env.step(1)  # Action 1: Add comment
        
        # Check step results
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (1000,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(self.env.steps, 1)
        self.assertIn("steps", info)
        self.assertIn("code_length", info)
        
    def test_action_space(self):
        """Test action space"""
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 5)
        
    def test_observation_space(self):
        """Test observation space"""
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        self.assertEqual(self.env.observation_space.shape, (1000,))
        self.assertEqual(self.env.observation_space.dtype, np.float32)
        
    def test_code_modifications(self):
        """Test different code modifications"""
        # Test each action
        actions = {
            0: lambda code: code,  # No change
            1: lambda code: code + "\n# Optimized",  # Add comment
            2: lambda code: code.replace("    ", "  "),  # Change indentation
            3: lambda code: code.upper(),  # Convert to uppercase
            4: lambda code: code.lower(),  # Convert to lowercase
        }
        
        for action, expected_mod in actions.items():
            self.env.reset()
            self.env.step(action)
            expected_code = expected_mod(self.initial_code)
            self.assertEqual(self.env.current_code, expected_code)
            
    def test_episode_termination(self):
        """Test episode termination"""
        # Run until max steps
        for _ in range(self.env.max_steps):
            state, reward, terminated, truncated, info = self.env.step(0)
            
        # Check termination
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(self.env.steps, self.env.max_steps)

if __name__ == '__main__':
    unittest.main() 