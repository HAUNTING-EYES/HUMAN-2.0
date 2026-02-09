import pytest
from src.components.code_actions import AdvancedCodeModifier, CodeAction
import ast
import libcst

@pytest.fixture
def code_modifier():
    return AdvancedCodeModifier()

@pytest.fixture
def sample_code():
    return '''
def process_data(data):
    # Repeated code block 1
    result1 = []
    for item in data:
        if item > 0:
            result1.append(item * 2)
    
    # Some other operations
    temp = sum(result1)
    
    # Repeated code block 2
    result2 = []
    for item in data:
        if item > 0:
            result2.append(item * 2)
    
    return result1, result2
'''

def test_initialize_actions(code_modifier):
    """Test that actions are properly initialized."""
    actions = code_modifier.actions
    assert len(actions) == 5
    assert all(isinstance(action, CodeAction) for action in actions)
    assert "extract_method" in [action.name for action in actions]

def test_extract_method(code_modifier, sample_code):
    """Test method extraction for repeated code."""
    result = code_modifier.extract_method(sample_code)
    
    # Check if a new method was created
    assert "def process_items" in result.modified_code
    # Check if repeated code was replaced with method calls
    assert result.modified_code.count("process_items(") >= 2
    # Verify the extracted method contains the common logic
    assert "append(item * 2)" in result.modified_code

def test_introduce_design_pattern(code_modifier):
    """Test introduction of design patterns."""
    code = '''
class DataProcessor:
    def process_text(self, text): pass
    def process_json(self, json): pass
    def process_xml(self, xml): pass
'''
    result = code_modifier.introduce_design_pattern(code, pattern="strategy")
    
    # Check for strategy pattern implementation
    assert "class ProcessingStrategy" in result.modified_code
    assert "class TextStrategy" in result.modified_code
    assert "def execute" in result.modified_code

def test_optimize_data_structures(code_modifier):
    """Test data structure optimization."""
    code = '''
def find_item(items, target):
    for item in items:
        if item == target:
            return True
    return False
'''
    result = code_modifier.optimize_data_structures(code)
    
    # Check if list was converted to set for O(1) lookup
    assert "set(" in result.modified_code
    assert "in items" in result.modified_code

def test_add_concurrency(code_modifier):
    """Test adding concurrent execution."""
    code = '''
def process_items(items):
    results = []
    for item in items:
        results.append(expensive_operation(item))
    return results
'''
    result = code_modifier.add_concurrency(code)
    
    # Check for concurrent execution implementation
    assert "concurrent.futures" in result.modified_code
    assert "ThreadPoolExecutor" in result.modified_code
    assert "map" in result.modified_code

def test_improve_error_handling(code_modifier):
    """Test error handling improvement."""
    code = '''
def process_data(data):
    return data['key']['nested']['value']
'''
    result = code_modifier.improve_error_handling(code)
    
    # Check for proper error handling
    assert "try:" in result.modified_code
    assert "except KeyError" in result.modified_code
    assert "raise" in result.modified_code
    assert "return" in result.modified_code

def test_invalid_code(code_modifier):
    """Test handling of invalid code."""
    invalid_code = "def invalid_syntax:"
    with pytest.raises(SyntaxError):
        code_modifier.extract_method(invalid_code)

def test_no_optimization_needed(code_modifier):
    """Test when no optimization is needed."""
    optimal_code = '''
def process_item(item):
    try:
        return item.process()
    except AttributeError as e:
        raise ValueError("Invalid item") from e
'''
    result = code_modifier.improve_error_handling(optimal_code)
    assert result.modified_code == optimal_code
    assert not result.changes_made

def test_multiple_optimizations(code_modifier, sample_code):
    """Test applying multiple optimizations."""
    # Apply multiple optimizations
    result1 = code_modifier.extract_method(sample_code)
    result2 = code_modifier.improve_error_handling(result1.modified_code)
    result3 = code_modifier.add_concurrency(result2.modified_code)
    
    # Verify all optimizations were applied
    assert "def process_items" in result3.modified_code  # Method extraction
    assert "try:" in result3.modified_code  # Error handling
    assert "ThreadPoolExecutor" in result3.modified_code  # Concurrency

def test_code_action_creation():
    """Test CodeAction dataclass."""
    action = CodeAction(
        name="extract_method",
        description="Extract repeated code",
        priority=1
    )
    assert action.name == "extract_method"
    assert action.priority == 1

def test_parameter_analysis(code_modifier):
    """Test parameter analysis functionality."""
    code = '''
def process_data(data: list, threshold: int = 0) -> list:
    return [x for x in data if x > threshold]
'''
    params = code_modifier._analyze_parameters(code)
    assert "data" in params
    assert "threshold" in params
    assert params["data"] == "list"
    assert params["threshold"] == "int"

def test_analyze_block_parameters(code_modifier):
    """Test parameter analysis."""
    blocks = [
        "result = x + y",
        "total = x + y"
    ]
    params = code_modifier._analyze_block_parameters(blocks)
    assert 'x' in params
    assert 'y' in params

def test_generate_method_name(code_modifier):
    """Test method name generation."""
    blocks = [
        "calculate_total_price(items)",
        "calculate_final_price(items)"
    ]
    name = code_modifier._generate_method_name(blocks)
    assert len(name) > 0
    assert '_' in name
    assert name.islower()

def test_extract_common_words(code_modifier):
    """Test common word extraction."""
    blocks = [
        "calculate_total_price(items)",
        "calculate_final_price(items)"
    ]
    words = code_modifier._extract_common_words(blocks)
    assert 'calculate' in words
    assert 'price' in words 