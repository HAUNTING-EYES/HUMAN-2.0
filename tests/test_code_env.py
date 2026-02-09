import pytest
import numpy as np
import gymnasium as gym
from src.components.code_env import CodeOptimizationEnv, CodeState

@pytest.fixture
def sample_code():
    return '''
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
'''

@pytest.fixture
def env(sample_code):
    return CodeOptimizationEnv(sample_code)

def test_env_initialization(env):
    """Test environment initialization."""
    assert env.action_space.n == 5  # Number of actions from AdvancedCodeModifier
    assert isinstance(env.observation_space.spaces['code_embedding'], gym.spaces.Box)
    assert isinstance(env.observation_space.spaces['metrics'], gym.spaces.Box)
    assert isinstance(env.observation_space.spaces['action_history'], gym.spaces.MultiDiscrete)

def test_reset(env):
    """Test environment reset."""
    observation = env.reset()
    assert 'code_embedding' in observation
    assert 'metrics' in observation
    assert 'action_history' in observation
    assert env.current_state is not None
    assert isinstance(env.current_state, CodeState)

def test_step(env):
    """Test environment step."""
    env.reset()
    observation, reward, done, truncated, info = env.step(0)  # Try extract_method action
    
    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'metrics' in info

def test_calculate_metrics(env, sample_code):
    """Test code metrics calculation."""
    metrics = env._calculate_metrics(sample_code)
    assert 'complexity' in metrics
    assert 'maintainability' in metrics
    assert 'performance' in metrics
    assert 'reliability' in metrics
    assert 'security' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

def test_calculate_complexity(env, sample_code):
    """Test complexity calculation."""
    complexity = env._calculate_complexity(sample_code)
    assert isinstance(complexity, float)
    assert complexity > 0  # Should have some complexity due to for loop

def test_calculate_maintainability(env, sample_code):
    """Test maintainability calculation."""
    maintainability = env._calculate_maintainability(sample_code)
    assert isinstance(maintainability, float)
    assert 0 <= maintainability <= 100  # Should be normalized

def test_estimate_performance(env, sample_code):
    """Test performance estimation."""
    performance = env._estimate_performance(sample_code)
    assert isinstance(performance, float)
    assert 0 < performance <= 1  # Should be normalized

def test_calculate_reliability(env, sample_code):
    """Test reliability calculation."""
    reliability = env._calculate_reliability(sample_code)
    assert isinstance(reliability, float)
    assert 0 < reliability <= 1  # Should be normalized

def test_analyze_security(env):
    """Test security analysis."""
    risky_code = '''
def unsafe_operation():
    user_input = input()
    eval(user_input)
'''
    security_score = env._analyze_security(risky_code)
    assert isinstance(security_score, float)
    assert security_score < 1  # Should be penalized for eval()

def test_calculate_reward(env):
    """Test reward calculation."""
    metrics1 = {
        'complexity': 1.0,
        'maintainability': 0.8,
        'performance': 0.9,
        'reliability': 0.7,
        'security': 1.0
    }
    metrics2 = {
        'complexity': 0.8,  # Lower complexity is better
        'maintainability': 0.9,
        'performance': 0.95,
        'reliability': 0.8,
        'security': 1.0
    }
    
    # First call should return 0 and set previous metrics
    reward1 = env._calculate_reward(metrics1)
    assert reward1 == 0.0
    
    # Second call should return positive reward for improvements
    reward2 = env._calculate_reward(metrics2)
    assert reward2 > 0

def test_is_done(env):
    """Test episode termination conditions."""
    env.reset()
    
    # Should not be done initially
    assert not env._is_done()
    
    # Should be done after max steps
    for _ in range(10):
        env.current_state.action_history.append("action")
    assert env._is_done()

def test_get_observation(env):
    """Test observation generation."""
    env.reset()
    observation = env._get_observation()
    
    assert 'code_embedding' in observation
    assert 'metrics' in observation
    assert 'action_history' in observation
    
    assert observation['code_embedding'].shape == (512,)
    assert observation['metrics'].shape == (5,)
    assert observation['action_history'].shape == (5,)

def test_embed_code(env, sample_code):
    """Test code embedding."""
    embedding = env._embed_code(sample_code)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert np.all(embedding >= 0)  # Should be normalized

def test_find_similar_blocks(env):
    """Test similar code block detection."""
    code_with_duplicates = '''
def func1(x):
    result = x * 2
    print(result)
    return result

def func2(y):
    result = y * 2
    print(result)
    return result
'''
    similar_blocks = env._find_similar_blocks(code_with_duplicates)
    assert len(similar_blocks) > 0
    # The similar block should contain the common lines with proper indentation
    expected_lines = [
        '    print(result)',
        '    return result'
    ]
    assert all(line in similar_blocks[0] for line in expected_lines)

def test_detect_appropriate_pattern(env):
    """Test design pattern detection."""
    factory_code = '''
def create_user(name):
    return User(name)

def build_profile(user):
    return Profile(user)
'''
    pattern = env._detect_appropriate_pattern(factory_code)
    assert pattern == 'factory'  # Should detect factory pattern 