import os
import pytest
import logging
import json
import numpy as np
from components.web_learning import WebLearningSystem
from components.self_improvement import SelfImprovementSystem

def configure_logging():
    """Configure logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test.log')
        ]
    )
    return logging.getLogger(__name__)

@pytest.fixture
def web_learning_system():
    """Fixture to create a WebLearningSystem instance"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return WebLearningSystem(base_dir)

@pytest.fixture
def sample_python_code():
    """Fixture to provide sample Python code for testing"""
    return '''
def example_function(param1, param2):
    """Example function with docstring."""
    result = param1 + param2
    return result

class ExampleClass:
    """Example class with docstring."""
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
'''

@pytest.fixture
def sample_markdown():
    """Fixture to provide sample markdown for testing"""
    return '''
# Example Markdown

This is a sample markdown file with some content.

## Features
- Feature 1
- Feature 2

[Link to something](https://example.com)
'''

def test_web_learning_initialization(web_learning_system):
    """Test WebLearningSystem initialization"""
    assert web_learning_system.base_dir is not None
    assert web_learning_system.visited_urls == set()
    assert web_learning_system.learning_data == []
    assert web_learning_system.headers is not None
    assert web_learning_system.dependency_graph is not None
    assert web_learning_system.cache_dir is not None
    assert web_learning_system.cache == {}
    assert web_learning_system.last_request_time == 0
    assert web_learning_system.min_request_interval == 1.0

def test_github_token_setting(web_learning_system):
    """Test GitHub token setting"""
    test_token = "test_token"
    web_learning_system.set_github_token(test_token)
    assert web_learning_system.github_client is not None
    assert web_learning_system.headers['Authorization'] == f'token {test_token}'

def test_file_priority_calculation(web_learning_system):
    """Test file priority calculation"""
    assert web_learning_system._get_file_priority('readme.md') > 0
    assert web_learning_system._get_file_priority('requirements.txt') > 0
    assert web_learning_system._get_file_priority('test_file.py') > 0
    assert web_learning_system._get_file_priority('core/main.py') > web_learning_system._get_file_priority('test_file.py')

def test_python_file_analysis(web_learning_system, sample_python_code):
    """Test Python file content analysis"""
    analysis = web_learning_system._analyze_file_content(sample_python_code, 'py')
    
    # Basic analysis checks
    assert 'imports' in analysis
    assert 'functions' in analysis
    assert 'classes' in analysis
    assert 'docstrings' in analysis
    assert 'complexity' in analysis
    assert 'nlp_analysis' in analysis
    
    # Function analysis checks
    assert 'example_function' in analysis['functions']
    assert len(analysis['functions']) == 1
    
    # Class analysis checks
    assert 'ExampleClass' in analysis['classes']
    assert len(analysis['classes']) == 1
    
    # Docstring checks
    assert len(analysis['docstrings']) == 2
    assert 'Example function with docstring' in analysis['docstrings'][0]
    
    # NLP analysis checks
    nlp_analysis = analysis['nlp_analysis']
    assert 'sentiment' in nlp_analysis
    assert 'key_phrases' in nlp_analysis
    assert 'semantic_embeddings' in nlp_analysis
    assert 'topic_keywords' in nlp_analysis
    assert 'code_quality_metrics' in nlp_analysis
    assert 'documentation_quality' in nlp_analysis
    
    # Code quality metrics checks
    metrics = nlp_analysis['code_quality_metrics']
    assert metrics['function_count'] == 1
    assert metrics['class_count'] == 1
    assert metrics['docstring_count'] == 2
    assert metrics['avg_function_length'] > 0

def test_markdown_analysis(web_learning_system, sample_markdown):
    """Test markdown file content analysis"""
    analysis = web_learning_system._analyze_file_content(sample_markdown, 'md')
    
    # Basic analysis checks
    assert 'headers' in analysis
    assert 'links' in analysis
    assert 'nlp_analysis' in analysis
    
    # Header checks
    assert len(analysis['headers']) == 2
    assert 'Example Markdown' in analysis['headers']
    
    # Link checks
    assert len(analysis['links']) == 1
    assert 'Link to something' in analysis['links'][0][0]
    
    # NLP analysis checks
    nlp_analysis = analysis['nlp_analysis']
    assert 'sentiment' in nlp_analysis
    assert 'key_phrases' in nlp_analysis
    assert 'topic_keywords' in nlp_analysis

def test_dependency_graph_building(web_learning_system):
    """Test dependency graph building"""
    files = [
        {
            'path': 'file1.py',
            'analysis': {
                'imports': ['numpy'],
                'functions': ['func1']
            }
        },
        {
            'path': 'file2.py',
            'analysis': {
                'imports': ['pandas'],
                'functions': ['func2']
            }
        }
    ]
    
    web_learning_system._build_dependency_graph(files)
    assert web_learning_system.dependency_graph.number_of_nodes() == 2
    assert web_learning_system.dependency_graph.number_of_edges() == 0

def test_web_learning_caching(web_learning_system):
    """Test web learning caching functionality"""
    test_data = {'test': 'data'}
    test_key = 'test_key'
    
    # Test saving to cache
    web_learning_system._save_to_cache(test_key, test_data)
    assert os.path.exists(os.path.join(web_learning_system.cache_dir, f"{test_key}.json"))
    
    # Test reading from cache
    cached_data = web_learning_system._get_cached_data(test_key)
    assert cached_data == test_data

def test_rate_limiting(web_learning_system):
    """Test rate limiting functionality"""
    import time
    
    start_time = time.time()
    web_learning_system._rate_limit()
    web_learning_system._rate_limit()
    end_time = time.time()
    
    # Should have waited at least min_request_interval seconds
    assert end_time - start_time >= web_learning_system.min_request_interval

def test_self_improvement_initialization():
    """Test SelfImprovementSystem initialization"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system = SelfImprovementSystem(base_dir)
    
    assert system.base_dir == base_dir
    assert system.web_learning is not None
    assert system.code_analyzer is not None
    assert system.continuous_learning is not None
    assert system.improvement_history == []
    assert system.current_improvements == []

def test_improvement_plan_generation():
    """Test improvement plan generation"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system = SelfImprovementSystem(base_dir)
    
    improvements = system._generate_improvement_plan()
    assert isinstance(improvements, list)
    
    # Check improvement structure
    for improvement in improvements:
        assert 'type' in improvement
        assert 'description' in improvement
        assert 'priority' in improvement

def test_improvement_execution():
    """Test improvement execution"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system = SelfImprovementSystem(base_dir)
    
    test_improvements = [
        {
            'type': 'code_optimization',
            'description': 'Test improvement',
            'priority': 'high'
        }
    ]
    
    system._execute_improvements(test_improvements)
    assert len(system.improvement_history) > 0
    assert system.improvement_history[0]['type'] == 'code_optimization'

def test_improvement_evaluation():
    """Test improvement evaluation"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system = SelfImprovementSystem(base_dir)
    
    # Add some test improvements
    system.improvement_history = [
        {'status': 'completed'},
        {'status': 'completed'},
        {'status': 'failed'}
    ]
    
    system._evaluate_improvements()
    evaluation_file = os.path.join(base_dir, 'improvement_evaluation.json')
    assert os.path.exists(evaluation_file)
    
    with open(evaluation_file, 'r') as f:
        evaluation = json.load(f)
        assert evaluation['total_improvements'] == 3
        assert evaluation['successful_improvements'] == 2
        assert evaluation['failed_improvements'] == 1
        assert evaluation['average_success_rate'] == pytest.approx(66.67, rel=1e-2)

def test_full_improvement_process():
    """Test the complete improvement process"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system = SelfImprovementSystem(base_dir)
    
    # Start the improvement process
    system.start_autonomous_improvement()
    
    # Verify results
    status = system.get_improvement_status()
    assert status['total_improvements'] >= 0
    assert status['current_improvements'] >= 0
    assert 0 <= status['success_rate'] <= 100
    
    # Check for generated files
    assert os.path.exists(os.path.join(base_dir, 'analysis_results.json'))
    assert os.path.exists(os.path.join(base_dir, 'improvement_evaluation.json'))
    assert os.path.exists(os.path.join(base_dir, 'learning_data'))

if __name__ == "__main__":
    pytest.main([__file__, '-v']) 