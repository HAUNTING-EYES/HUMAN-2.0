import unittest
import asyncio
import time
import numpy as np
from src.consciousness.self_awareness import SelfAwarenessSystem, AttentionFocus, ProcessState
from src.consciousness.reflection import ReflectionEngine, ReflectionTrigger
from src.consciousness.curiosity import CuriosityEngine, KnowledgeDomain
from src.consciousness.physiology import PhysiologicalSystem

class TestConsciousnessFramework(unittest.TestCase):
    def setUp(self):
        """Initialize all consciousness subsystems"""
        self.self_awareness = SelfAwarenessSystem()
        self.reflection = ReflectionEngine()
        self.curiosity = CuriosityEngine()
        self.physiology = PhysiologicalSystem()

    def test_self_awareness_state_tracking(self):
        """Test if self-awareness system can track internal states"""
        # Add a test goal
        test_goal = {"id": "test_goal", "priority": 0.8, "description": "Test goal"}
        self.self_awareness.current_goals.append(test_goal)

        # Add a test process
        current_time = time.time()
        test_process = ProcessState(
            process_id="test_process",
            status="running",
            resource_usage={"cpu": 0.5, "memory": 0.3},
            start_time=current_time,
            last_update=current_time
        )
        self.self_awareness.active_processes["test_process"] = test_process

        # Update state
        self.self_awareness.update_state()

        # Get current state
        state = self.self_awareness.get_current_state()

        # Verify state tracking
        self.assertIn("test_goal", [g["id"] for g in state["goals"]])
        self.assertIn("test_process", state["processes"])
        self.assertIsInstance(state["attention"], str)

    def test_reflection_trigger_response(self):
        """Test if reflection system responds to triggers"""
        # Create a test experience
        test_experience = {
            "type": "unexpected_outcome",
            "content": {
                "expected": "success",
                "actual": "failure",
                "context": "test operation"
            },
            "timestamp": time.time()
        }

        # Process experience (async)
        asyncio.run(self.reflection.process_experience(test_experience))

        # Check if reflection was triggered
        self.assertIsNotNone(self.reflection.current_reflection)
        self.assertIn("triggers", self.reflection.current_reflection)

    def test_curiosity_knowledge_update(self):
        """Test if curiosity system updates knowledge and generates questions"""
        # Add new knowledge
        test_knowledge = {
            "concept": "consciousness",
            "domain": "cognitive",
            "confidence": 0.7,
            "related_concepts": ["awareness", "attention"]
        }
        self.curiosity.update_knowledge(test_knowledge)

        # Generate questions
        questions = self.curiosity.generate_curiosity()

        # Verify question generation
        self.assertTrue(len(questions) > 0)
        self.assertTrue(any(q.domain == KnowledgeDomain.COGNITIVE for q in questions))

    def test_physiological_emotional_response(self):
        """Test if physiological system responds to emotional input"""
        # Initial state
        initial_state = self.physiology.get_state()

        # Simulate joy emotion
        emotional_state = {"joy": 0.8}
        self.physiology.update(emotional_state)

        # Get updated state
        updated_state = self.physiology.get_state()

        # Verify physiological changes
        self.assertNotEqual(
            initial_state["dopamine"],
            updated_state["dopamine"],
            "Dopamine level should change in response to joy"
        )

    def test_integration_emotional_reflection(self):
        """Test integration between emotional and reflection systems"""
        # Create emotional experience
        emotional_exp = {
            "type": "emotional_event",
            "content": {
                "emotion": "joy",
                "intensity": 0.8,
                "trigger": "achievement"
            },
            "timestamp": time.time()
        }

        # Process through reflection
        self.reflection.process_experience(emotional_exp)

        # Update physiology
        self.physiology.update({"joy": 0.8})

        # Verify reflection response
        self.assertIsNotNone(self.reflection.current_reflection)
        self.assertTrue(
            any("emotional" in str(trigger).lower() 
                for trigger in self.reflection.current_reflection.get("triggers", []))
        )

    def test_attention_switching(self):
        """Test if attention can switch based on stimuli"""
        # Initial focus
        initial_focus = self.self_awareness.attention_focus

        # Simulate high-priority external input
        test_input = {
            "type": "external_stimulus",
            "priority": 0.9,
            "content": "urgent test input"
        }
        self.self_awareness.add_experience(test_input)
        self.self_awareness.update_state()

        # Verify attention switch
        self.assertEqual(
            self.self_awareness.attention_focus,
            AttentionFocus.EXTERNAL,
            "Attention should switch to external focus for high-priority input"
        )

    def test_curiosity_learning_cycle(self):
        """Test the complete curiosity learning cycle"""
        # Add initial knowledge
        initial_knowledge = {
            "concept": "test_concept",
            "domain": "cognitive",
            "confidence": 0.5
        }
        self.curiosity.update_knowledge(initial_knowledge)

        # Generate questions
        questions = self.curiosity.generate_curiosity()
        initial_question_count = len(questions)

        # Add new related knowledge
        new_knowledge = {
            "concept": "related_concept",
            "domain": "cognitive",
            "confidence": 0.8,
            "related_to": "test_concept"
        }
        self.curiosity.update_knowledge(new_knowledge)

        # Generate new questions
        new_questions = self.curiosity.generate_curiosity()

        # Verify learning cycle
        self.assertNotEqual(
            initial_question_count,
            len(new_questions),
            "Question generation should adapt to new knowledge"
        )

    def test_physiological_stability(self):
        """Test physiological system's stability and homeostasis"""
        # Get initial stability
        initial_metrics = self.physiology.get_stability_metrics()

        # Introduce strong emotion
        self.physiology.update({"fear": 0.9})
        
        # Get immediate response metrics
        stress_metrics = self.physiology.get_stability_metrics()
        
        # Let system stabilize (simulate time passage)
        for _ in range(5):
            self.physiology.update({}, delta_time=1.0)
            
        # Get final stability metrics
        final_metrics = self.physiology.get_stability_metrics()

        # Verify homeostatic tendency
        self.assertLess(
            stress_metrics["overall"],
            initial_metrics["overall"],
            "Stress should decrease stability"
        )
        self.assertGreater(
            final_metrics["overall"],
            stress_metrics["overall"],
            "System should tend toward stability"
        )

if __name__ == '__main__':
    unittest.main() 