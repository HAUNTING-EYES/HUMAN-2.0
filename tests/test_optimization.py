import pytest
from src.optimization.pipeline import OptimizationPipeline
from src.components.code_actions import CodeAction
from dataclasses import dataclass
from typing import List, Dict

@pytest.fixture
def sample_code():
    return '''
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total
'''

@pytest.fixture
def optimization_pipeline():
    return OptimizationPipeline()

def test_pipeline_initialization(optimization_pipeline):
    """Test pipeline initialization."""
    assert optimization_pipeline.actions is not None
    assert len(optimization_pipeline.actions) > 0

def test_code_analysis(optimization_pipeline, sample_code):
    """Test code analysis step."""
    analysis = optimization_pipeline.analyze_code(sample_code)
    assert isinstance(analysis, dict)
    assert "ast" in analysis
    assert "metrics" in analysis
    assert "potential_improvements" in analysis

def test_action_selection(optimization_pipeline, sample_code):
    """Test action selection based on code analysis."""
    analysis = optimization_pipeline.analyze_code(sample_code)
    actions = optimization_pipeline.select_actions(analysis)
    assert isinstance(actions, list)
    assert all(isinstance(action, CodeAction) for action in actions)

def test_action_application(optimization_pipeline, sample_code):
    """Test applying selected actions to code."""
    result = optimization_pipeline.apply_actions(sample_code)
    assert isinstance(result.optimized_code, str)
    assert isinstance(result.improvements, list)
    assert isinstance(result.metrics, dict)

def test_optimization_limits(optimization_pipeline, sample_code):
    """Test respecting optimization step limits."""
    result = optimization_pipeline.optimize(sample_code, max_steps=1)
    assert result.steps_taken <= 1

def test_optimization_types(optimization_pipeline, sample_code):
    """Test different optimization types."""
    # Test performance optimization
    perf_result = optimization_pipeline.optimize(
        sample_code, 
        optimization_type="performance"
    )
    assert "performance" in perf_result.metrics

    # Test readability optimization
    read_result = optimization_pipeline.optimize(
        sample_code, 
        optimization_type="readability"
    )
    assert "readability" in read_result.metrics

def test_code_metrics(optimization_pipeline, sample_code):
    """Test code quality metrics calculation."""
    metrics = optimization_pipeline.calculate_metrics(sample_code)
    assert "complexity" in metrics
    assert "maintainability" in metrics
    assert isinstance(metrics["complexity"], float)
    assert isinstance(metrics["maintainability"], float)

def test_invalid_optimization_type(optimization_pipeline, sample_code):
    """Test handling of invalid optimization type."""
    with pytest.raises(ValueError):
        optimization_pipeline.optimize(sample_code, optimization_type="invalid")

def test_empty_code(optimization_pipeline):
    """Test handling of empty code."""
    with pytest.raises(ValueError):
        optimization_pipeline.optimize("")

def test_optimization_improvement(optimization_pipeline, sample_code):
    """Test that optimization actually improves code metrics."""
    initial_metrics = optimization_pipeline.calculate_metrics(sample_code)
    result = optimization_pipeline.optimize(sample_code)
    final_metrics = optimization_pipeline.calculate_metrics(result.optimized_code)
    
    # At least one metric should improve
    assert any(
        final_metrics[key] > initial_metrics[key] 
        for key in initial_metrics 
        if key in final_metrics
    )

def test_concurrent_optimization(optimization_pipeline):
    """Test concurrent optimization of multiple code snippets."""
    code_snippets = [
        "def f1(x): return x + 1",
        "def f2(x): return x * 2",
        "def f3(x): return x ** 2"
    ]
    
    results = optimization_pipeline.optimize_batch(code_snippets)
    assert len(results) == len(code_snippets)
    assert all(hasattr(r, 'optimized_code') for r in results) 