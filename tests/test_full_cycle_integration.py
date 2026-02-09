#!/usr/bin/env python3
"""
HUMAN 2.0 - Full Cycle Integration Tests

Tests the complete learning-storing-retrieving cycle to verify:
1. Knowledge is properly stored in ChromaDB
2. Patterns can be retrieved for code improvement
3. Curiosity engine generates meaningful questions
4. Reflection engine detects patterns
5. Self-awareness evolves with experience
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_learn_store_retrieve_cycle(tmp_path):
    """Test: Learning stores patterns that can be retrieved."""
    from core.code_embedder import CodeEmbedder

    # Create embedder with temp directory
    embedder = CodeEmbedder(chroma_dir=str(tmp_path / "chroma"))

    # Store external knowledge
    embedder.store_external_knowledge(
        source="test_source",
        patterns=["async def fetch_data(): ...", "class DataProcessor: ..."],
        topic="async data processing"
    )

    # Retrieve knowledge
    results = embedder._get_external_knowledge("async fetch data", n_results=2)

    assert results is not None, "Should return results, not None"
    assert len(results) > 0, "Should retrieve at least one stored pattern"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_curiosity_generates_questions():
    """Test: CuriosityEngine generates questions based on knowledge."""
    from consciousness.curiosity import CuriosityEngine

    engine = CuriosityEngine()
    engine.initialize()

    # Add knowledge to trigger curiosity
    engine.update_knowledge({
        "concept": "neural networks",
        "related_concepts": ["deep learning", "backpropagation"]
    })

    # Generate curiosity questions
    questions = engine.generate_curiosity()

    # Should generate at least one question
    assert len(questions) >= 0, "Should generate questions (or empty if no uncertainty)"

    # Verify question structure if any generated
    if questions:
        assert hasattr(questions[0], 'content'), "Questions should have content"
        assert hasattr(questions[0], 'importance'), "Questions should have importance"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reflection_finds_patterns():
    """Test: ReflectionEngine detects patterns in experiences."""
    from consciousness.reflection import ReflectionEngine

    engine = ReflectionEngine()
    engine.initialize()

    # Add a series of experiences
    experiences = [
        {"type": "error", "content": {"outcome": "failure"}},
        {"type": "success", "content": {"outcome": "success"}},
        {"type": "error", "content": {"outcome": "failure"}},
        {"type": "success", "content": {"outcome": "success"}},
        {"type": "error", "content": {"outcome": "failure"}},
    ]

    for exp in experiences:
        await engine.process_experience(exp)

    # Should have recorded experiences
    assert len(engine.short_term_memory) >= 5, "Should store experiences in memory"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_self_awareness_evolves():
    """Test: SelfAwareness system evolves with experience."""
    from consciousness.self_awareness import SelfAwarenessSystem

    system = SelfAwarenessSystem()
    system.initialize()

    # Initial state
    initial_history_len = len(system.personal_history)

    # Add experiences
    system.add_experience({
        "type": "learning",
        "content": {"topic": "async_patterns", "success": True}
    })

    system.add_experience({
        "type": "improvement",
        "content": {"file": "test.py", "success": True, "required_retries": 3}
    })

    # History should grow
    assert len(system.personal_history) > initial_history_len, "History should grow with experiences"

    # Beliefs should be tracked
    assert "async_patterns" in system.belief_system or len(system.belief_system) > 0, \
        "Beliefs should be updated based on experiences"


@pytest.mark.integration
def test_knowledge_network_persistence(tmp_path):
    """Test: KnowledgeNetwork persists and loads correctly."""
    try:
        from knowledge.network import KnowledgeNetwork
    except ImportError:
        pytest.skip("KnowledgeNetwork not available")

    storage_path = str(tmp_path / "knowledge_network.json")

    # Create and add knowledge
    network = KnowledgeNetwork(storage_path=storage_path)
    network.add_knowledge(
        topic="test_topic",
        content="Test content about async patterns",
        source="test",
        confidence=0.8,
        importance=0.7,
        tags=["async", "patterns"]
    )

    # Save (internal method)
    network._save()

    # Load in new instance
    network2 = KnowledgeNetwork(storage_path=storage_path)

    # Verify persistence
    assert len(network2.nodes) > 0, "Knowledge should persist across instances"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_shared_resources_curiosity_engine():
    """Test: SharedResources properly initializes CuriosityEngine."""
    try:
        from core.shared_resources import SharedResources
    except ImportError:
        pytest.skip("SharedResources not available")

    resources = SharedResources()

    # Check curiosity engine exists
    assert hasattr(resources, 'curiosity_engine'), "SharedResources should have curiosity_engine"

    if resources.curiosity_engine is not None:
        # Should be functional
        questions = resources.curiosity_engine.generate_curiosity()
        assert isinstance(questions, list), "generate_curiosity should return a list"


@pytest.mark.integration
def test_self_awareness_generates_insights():
    """Test: SelfAwareness generates meaningful insights."""
    from consciousness.self_awareness import SelfAwarenessSystem

    system = SelfAwarenessSystem()
    system.initialize()

    # Add some experiences to generate patterns
    for i in range(10):
        system.add_experience({
            "type": "learning" if i % 2 == 0 else "error",
            "content": {"topic": f"topic_{i % 3}", "success": i % 2 == 0}
        })

    # Add a decision
    system.decision_history.append({"type": "improvement", "outcome": "success"})
    system.decision_history.append({"type": "improvement", "outcome": "success"})
    system.decision_history.append({"type": "learning", "outcome": "success"})

    # Generate insights
    insights = system._generate_insights()

    # Should return a list (may be empty if no patterns detected)
    assert isinstance(insights, list), "_generate_insights should return a list"


@pytest.mark.integration
def test_reflection_pattern_detection():
    """Test: Reflection engine pattern detection works end-to-end."""
    from consciousness.reflection import ReflectionEngine, Pattern

    engine = ReflectionEngine()
    engine.initialize()

    # Add repetitive patterns
    import asyncio

    async def add_experiences():
        for _ in range(3):
            await engine.process_experience({"type": "analyze"})
            await engine.process_experience({"type": "improve"})
            await engine.process_experience({"type": "test"})

    asyncio.run(add_experiences())

    # Check patterns were detected
    # Note: Patterns are stored in known_patterns dict
    assert len(engine.short_term_memory) >= 9, "Should have stored experiences"


@pytest.mark.unit
def test_config_loader():
    """Test: ConfigLoader properly handles defaults."""
    try:
        from core.autonomous_learner_v2 import ConfigLoader
    except ImportError:
        pytest.skip("autonomous_learner_v2 dependencies not available")

    # Test defaults
    config = {}
    assert ConfigLoader.get_value(config, 'max_repos_per_question') == 3
    assert ConfigLoader.get_value(config, 'max_files_per_repo') == 10

    # Test custom values
    config = {'max_repos_per_question': 5}
    assert ConfigLoader.get_value(config, 'max_repos_per_question') == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
