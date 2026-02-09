import unittest
import torch
import numpy as np
from src.components.emotional_scenarios import ComplexEmotionalScenarios
from pathlib import Path
import shutil
import tempfile

class TestComplexEmotionalScenarios(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.scenarios = ComplexEmotionalScenarios(base_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test initialization of the scenarios handler."""
        self.assertIsNotNone(self.scenarios.conflict_resolver)
        self.assertIsNotNone(self.scenarios.transition_manager)
        self.assertIsNotNone(self.scenarios.scenario_classifier)
        self.assertEqual(len(self.scenarios.scenario_history), 0)
        self.assertEqual(len(self.scenarios.active_scenarios), 0)
        
    def test_scenario_classification(self):
        """Test scenario classification."""
        # Test simple scenario
        simple_state = {
            'joy': 0.8,
            'sadness': 0.1,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        scenario_type = self.scenarios._classify_scenario(simple_state)
        self.assertIn(scenario_type, [
            'simple', 'conflict', 'transition', 'complex',
            'stable', 'unstable', 'intense', 'mild',
            'positive', 'negative'
        ])
        
        # Test conflict scenario
        conflict_state = {
            'joy': 0.8,
            'sadness': 0.7,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        scenario_type = self.scenarios._classify_scenario(conflict_state)
        self.assertIn(scenario_type, [
            'simple', 'conflict', 'transition', 'complex',
            'stable', 'unstable', 'intense', 'mild',
            'positive', 'negative'
        ])
        
    def test_conflict_detection(self):
        """Test conflict detection."""
        # Test conflicting emotions
        conflict_state = {
            'joy': 0.8,
            'sadness': 0.7,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        conflicts = self.scenarios._detect_conflicts(conflict_state)
        self.assertTrue(len(conflicts) > 0)
        
        # Test non-conflicting emotions
        non_conflict_state = {
            'joy': 0.8,
            'surprise': 0.7,
            'anger': 0.1,
            'fear': 0.1,
            'sadness': 0.1,
            'disgust': 0.1
        }
        conflicts = self.scenarios._detect_conflicts(non_conflict_state)
        self.assertEqual(len(conflicts), 0)
        
    def test_conflict_resolution(self):
        """Test conflict resolution."""
        initial_state = {
            'joy': 0.8,
            'sadness': 0.7,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        
        resolved_state = self.scenarios._resolve_conflicts(
            initial_state,
            self.scenarios._detect_conflicts(initial_state)
        )
        
        # Check that all emotions are present
        self.assertEqual(set(resolved_state.keys()), set(initial_state.keys()))
        
        # Check that values are within valid range
        for value in resolved_state.values():
            self.assertTrue(0 <= value <= 1)
            
    def test_transition_management(self):
        """Test transition management."""
        initial_state = {
            'joy': 0.8,
            'sadness': 0.1,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        
        target_state = {
            'joy': 0.1,
            'sadness': 0.8,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        
        context = {'target_emotional_state': target_state}
        
        transitioned_state = self.scenarios._manage_transitions(initial_state, context)
        
        # Check that all emotions are present
        self.assertEqual(set(transitioned_state.keys()), set(initial_state.keys()))
        
        # Check that values are within valid range
        for value in transitioned_state.values():
            self.assertTrue(0 <= value <= 1)
            
        # Check that transition is in the right direction
        self.assertTrue(transitioned_state['joy'] < initial_state['joy'])
        self.assertTrue(transitioned_state['sadness'] > initial_state['sadness'])
        
    def test_scenario_processing(self):
        """Test complete scenario processing."""
        initial_state = {
            'joy': 0.8,
            'sadness': 0.7,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        
        context = {
            'target_emotional_state': {
                'joy': 0.1,
                'sadness': 0.8,
                'anger': 0.1,
                'fear': 0.1,
                'surprise': 0.1,
                'disgust': 0.1
            }
        }
        
        processed_state = self.scenarios.process_scenario(initial_state, context)
        
        # Check that all emotions are present
        self.assertEqual(set(processed_state.keys()), set(initial_state.keys()))
        
        # Check that values are within valid range
        for value in processed_state.values():
            self.assertTrue(0 <= value <= 1)
            
        # Check that scenario was stored
        self.assertTrue(len(self.scenarios.scenario_history) > 0)
        
    def test_scenario_stats(self):
        """Test scenario statistics."""
        # Process some scenarios
        states = [
            {
                'joy': 0.8, 'sadness': 0.1, 'anger': 0.1,
                'fear': 0.1, 'surprise': 0.1, 'disgust': 0.1
            },
            {
                'joy': 0.1, 'sadness': 0.8, 'anger': 0.1,
                'fear': 0.1, 'surprise': 0.1, 'disgust': 0.1
            },
            {
                'joy': 0.8, 'sadness': 0.7, 'anger': 0.1,
                'fear': 0.1, 'surprise': 0.1, 'disgust': 0.1
            }
        ]
        
        for state in states:
            self.scenarios.process_scenario(state, {'target_emotional_state': state})
            
        stats = self.scenarios.get_scenario_stats()
        
        # Check basic stats
        self.assertEqual(stats['total_scenarios'], 3)
        self.assertTrue('scenario_types' in stats)
        self.assertTrue('conflict_stats' in stats)
        self.assertTrue('transition_stats' in stats)
        
    def test_scenario_storage(self):
        """Test scenario storage and retrieval."""
        # Process a scenario
        state = {
            'joy': 0.8,
            'sadness': 0.1,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        
        self.scenarios.process_scenario(state, {'target_emotional_state': state})
        
        # Check file was created
        scenario_file = Path(self.temp_dir) / "scenarios.json"
        self.assertTrue(scenario_file.exists())
        
        # Test loading scenarios
        new_scenarios = ComplexEmotionalScenarios(base_dir=self.temp_dir)
        self.assertTrue(len(new_scenarios.scenario_history) > 0)
        
    def test_recent_scenarios(self):
        """Test retrieval of recent scenarios."""
        # Process multiple scenarios
        states = [
            {
                'joy': 0.8, 'sadness': 0.1, 'anger': 0.1,
                'fear': 0.1, 'surprise': 0.1, 'disgust': 0.1
            },
            {
                'joy': 0.1, 'sadness': 0.8, 'anger': 0.1,
                'fear': 0.1, 'surprise': 0.1, 'disgust': 0.1
            },
            {
                'joy': 0.8, 'sadness': 0.7, 'anger': 0.1,
                'fear': 0.1, 'surprise': 0.1, 'disgust': 0.1
            }
        ]
        
        for state in states:
            self.scenarios.process_scenario(state, {'target_emotional_state': state})
            
        # Get recent scenarios
        recent = self.scenarios.get_recent_scenarios(count=2)
        self.assertEqual(len(recent), 2)
        
        # Check that most recent scenario has appropriate values
        last_state = recent[-1]['emotional_state']
        self.assertGreaterEqual(last_state['sadness'], 0.4)  # Should be moderately high
        self.assertLessEqual(last_state['sadness'], 0.9)     # But not too high
        self.assertGreaterEqual(last_state['joy'], 0.5)      # Should also be moderately high
        self.assertLessEqual(last_state['joy'], 0.9)         # But not too high
        
    def test_clear_scenarios(self):
        """Test clearing scenarios."""
        # Process a scenario
        state = {
            'joy': 0.8,
            'sadness': 0.1,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'disgust': 0.1
        }
        
        self.scenarios.process_scenario(state, {'target_emotional_state': state})
        
        # Clear scenarios
        self.scenarios.clear_scenarios()
        
        # Check that scenarios are cleared
        self.assertEqual(len(self.scenarios.scenario_history), 0)
        
        # Check that file is empty
        scenario_file = Path(self.temp_dir) / "scenarios.json"
        self.assertTrue(scenario_file.exists())
        with open(scenario_file, 'r') as f:
            content = f.read()
            self.assertEqual(content, '[]') 