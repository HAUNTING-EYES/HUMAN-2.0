import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from src.components.web_learning import WebLearningSystem

@pytest.fixture
def web_learning_system(tmp_path):
    base_dir = str(tmp_path / "learning_data")
    os.makedirs(base_dir, exist_ok=True)
    return WebLearningSystem(base_dir=base_dir)

@pytest.fixture
def mock_response():
    mock = Mock()
    mock.text = """
    <html>
        <body>
            <h1>Test Repository</h1>
            <div class="file-content">
                <pre><code>def test_function():
    return "Hello World"</code></pre>
            </div>
            <a href="https://github.com/test/repo/blob/main/test.py">test.py</a>
        </body>
    </html>
    """
    mock.json = Mock(side_effect=[
        {
            'name': 'test-repo',
            'description': 'Test repository',
            'stargazers_count': 10
        },
        [
            {
                'type': 'file',
                'path': 'test.py',
                'name': 'test.py',
                'download_url': 'https://raw.githubusercontent.com/test/repo/main/test.py'
            }
        ],
        "def test(): pass"
    ])
    mock.raise_for_status = Mock()
    return mock

@pytest.fixture
def mock_github_client():
    mock = Mock()
    mock_repo = Mock()
    
    # Mock repository attributes
    mock_repo.name = 'test-repo'
    mock_repo.description = 'Test repository'
    mock_repo.stargazers_count = 10
    
    # Create mock contents
    mock_py_file = Mock()
    mock_py_file.type = "file"
    mock_py_file.path = "test.py"
    mock_py_file.decoded_content = b"def test(): pass"
    mock_py_file.name = "test.py"
    
    mock_dir = Mock()
    mock_dir.type = "dir"
    mock_dir.path = "src"
    mock_dir.decoded_content = None
    mock_dir.name = "src"
    
    # Set up get_contents to return a list of contents
    mock_repo.get_contents.return_value = [mock_py_file, mock_dir]
    
    # Set up get_repo to return our mock repository
    mock.get_repo = Mock(return_value=mock_repo)
    
    # Mock the authentication
    mock_user = Mock()
    mock_user.login = 'test-user'
    mock.get_user = Mock(return_value=mock_user)
    
    return mock

def test_learn_from_github(web_learning_system, mock_response):
    with patch('requests.get', return_value=mock_response):
        result = web_learning_system.learn_from_github('https://github.com/test/repo')
        assert result['name'] == 'test-repo'
        assert result['description'] == 'Test repository'
        assert result['stars'] == 10

def test_learn_from_github_with_client(web_learning_system, mock_response, mock_github_client):
    with patch('requests.get', return_value=mock_response), \
         patch('github.Github', return_value=mock_github_client), \
         patch('github.Auth.Token', return_value=Mock()):
        web_learning_system.set_github_token('test-token')
        result = web_learning_system.learn_from_github('https://github.com/test/repo')
        assert result['name'] == 'test-repo'
        assert result['description'] == 'Test repository'
        assert result['stars'] == 10
        assert len(result['files']) > 0
        assert any(f['path'] == 'test.py' for f in result['files'])

def test_learn_from_github_invalid_url(web_learning_system):
    with pytest.raises(ValueError, match="Invalid GitHub URL"):
        web_learning_system.learn_from_github('invalid-url')

def test_learn_from_url(web_learning_system, mock_response):
    with patch('requests.get', return_value=mock_response):
        result = web_learning_system.learn_from_url('https://github.com/test/repo')
        assert result['name'] == 'test-repo'
        assert result['description'] == 'Test repository'
        assert result['stars'] == 10

def test_learn_from_url_error(web_learning_system):
    with patch('requests.get', side_effect=Exception('Network error')):
        result = web_learning_system.learn_from_url('https://example.com')
        assert 'error' in result
        assert result['error'] == 'Network error'

def test_analyze_learned_data(web_learning_system):
    web_learning_system.learning_data = [{
        'files': [{
            'type': 'py',
            'metadata': {
                'functions': ['test_function'],
                'classes': ['TestClass']
            }
        }]
    }]
    analysis = web_learning_system.analyze_learned_data()
    assert 'languages_used' in analysis
    assert 'python' in analysis['languages_used']
    assert 'common_patterns' in analysis
    assert analysis['common_patterns']['test_function'] == 1
    assert analysis['common_patterns']['TestClass'] == 1

def test_analyze_learned_data_empty(web_learning_system):
    web_learning_system.learning_data = []
    analysis = web_learning_system.analyze_learned_data()
    assert analysis['total_pages'] == 0
    assert len(analysis['languages_used']) == 0
    assert len(analysis['common_patterns']) == 0

def test_suggest_improvements(web_learning_system):
    web_learning_system.learning_data = [{
        'files': [{
            'type': 'py',
            'metadata': {
                'functions': ['test_function'],
                'classes': ['TestClass']
            }
        }]
    }]
    suggestions = web_learning_system.suggest_improvements()
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert all('type' in suggestion for suggestion in suggestions)
    assert all('description' in suggestion for suggestion in suggestions)
    assert all('priority' in suggestion for suggestion in suggestions)

def test_suggest_improvements_empty(web_learning_system):
    web_learning_system.learning_data = []
    suggestions = web_learning_system.suggest_improvements()
    assert len(suggestions) > 0  # Should still have basic suggestions
    assert all(s['type'] in ['code_quality', 'documentation', 'testing'] for s in suggestions)

def test_save_and_load_learning_data(web_learning_system, tmp_path):
    test_data = {
        'name': 'test-repo',
        'description': 'Test repository',
        'files': []
    }
    web_learning_system.learning_data = [test_data]
    web_learning_system._save_learning_data()
    
    # Create new instance to test loading
    new_system = WebLearningSystem(base_dir=str(tmp_path / "learning_data"))
    assert len(new_system.learning_data) > 0
    assert new_system.learning_data[0]['name'] == 'test-repo'

def test_save_and_load_learning_data_error(web_learning_system, tmp_path):
    # Test saving with invalid data
    web_learning_system.learning_data = [{'invalid': object()}]  # Non-serializable object
    web_learning_system._save_learning_data()  # Should handle error gracefully
    
    # Test loading from non-existent directory
    new_system = WebLearningSystem(base_dir=str(tmp_path / "nonexistent"))
    assert len(new_system.learning_data) == 0

def test_rate_limiting(web_learning_system):
    with patch('requests.get', side_effect=Exception('Rate limit exceeded')):
        result = web_learning_system.learn_from_github('https://github.com/test/repo')
        assert result == {}

def test_rate_limiting_delay(web_learning_system, mock_response):
    with patch('time.time') as mock_time, \
         patch('time.sleep') as mock_sleep, \
         patch('requests.get', return_value=mock_response):
        mock_time.side_effect = [0, 0.5]  # Simulate 0.5 seconds between requests
        web_learning_system._rate_limit()
        mock_sleep.assert_called_once_with(0.5)  # Should sleep for remaining time

def test_caching(web_learning_system, mock_response):
    with patch('requests.get', return_value=mock_response) as mock_get:
        # First call should make a request
        result1 = web_learning_system.learn_from_github('https://github.com/test/repo')
        initial_calls = mock_get.call_count
        
        # Second call should use cache
        result2 = web_learning_system.learn_from_github('https://github.com/test/repo')
        assert mock_get.call_count == initial_calls  # No additional calls should be made
        assert result1 == result2

def test_cache_invalidation(web_learning_system, mock_response):
    with patch('requests.get', return_value=mock_response) as mock_get:
        # First call
        result1 = web_learning_system.learn_from_github('https://github.com/test/repo')
        assert result1['name'] == 'test-repo'
        
        # Reset mock call count and response
        mock_get.reset_mock()
        mock_get.return_value = Mock()
        mock_get.return_value.json = Mock(side_effect=[
            {
                'name': 'test-repo',
                'description': 'Test repository',
                'stargazers_count': 10
            },
            [
                {
                    'type': 'file',
                    'path': 'test.py',
                    'name': 'test.py',
                    'download_url': 'https://raw.githubusercontent.com/test/repo/main/test.py'
                }
            ],
            "def test(): pass"
        ])
        mock_get.return_value.raise_for_status = Mock()
        
        # Simulate cache file deletion
        cache_file = os.path.join(web_learning_system.cache_dir, 'github_test_repo.json')
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        # Should make a new request
        result2 = web_learning_system.learn_from_github('https://github.com/test/repo')
        assert result2['name'] == 'test-repo'
        assert mock_get.call_count > 0

def test_dependency_graph(web_learning_system):
    files = [
        {
            'path': 'test.py',
            'analysis': {
                'imports': ['module1'],
                'functions': ['test_function']
            }
        },
        {
            'path': 'module1.py',
            'analysis': {
                'functions': ['module1']
            }
        }
    ]
    web_learning_system._build_dependency_graph(files)
    assert web_learning_system.dependency_graph.number_of_nodes() == 2
    assert web_learning_system.dependency_graph.number_of_edges() == 1

def test_dependency_graph_empty(web_learning_system):
    web_learning_system._build_dependency_graph([])
    assert web_learning_system.dependency_graph.number_of_nodes() == 0
    assert web_learning_system.dependency_graph.number_of_edges() == 0

def test_file_priority(web_learning_system):
    priorities = {
        'requirements.txt': 100,  # High priority file
        'setup.py': 100,  # High priority file
        'README.md': 100,  # High priority file
        'src/test.py': 110,  # 50 for .py + 40 for src/ + 20 for test
        'docs/api.md': 30  # 30 for .md
    }
    
    for filename, expected_priority in priorities.items():
        priority = web_learning_system._get_file_priority(filename)
        assert priority == expected_priority, f"Priority mismatch for {filename}: expected {expected_priority}, got {priority}"

def test_file_priority_edge_cases(web_learning_system):
    # Test with empty filename
    assert web_learning_system._get_file_priority('') == 0
    
    # Test with no extension
    assert web_learning_system._get_file_priority('file') == 0
    
    # Test with multiple extensions
    assert web_learning_system._get_file_priority('file.py.txt') == 50  # Should use first extension
    
    # Test with uppercase extensions
    assert web_learning_system._get_file_priority('file.PY') == 50
    
    # Test with mixed case paths
    assert web_learning_system._get_file_priority('SRC/test.py') == 110

def test_github_token_setting(web_learning_system):
    mock_client = Mock()
    mock_user = Mock()
    mock_user.login = 'test-user'
    mock_client.get_user = Mock(return_value=mock_user)
    mock_auth = Mock()
    
    with patch('github.Github', return_value=mock_client) as mock_github, \
         patch('github.Auth.Token', return_value=mock_auth) as mock_auth_token:
        web_learning_system.set_github_token('test-token')
        assert mock_auth_token.called
        assert mock_github.called
        assert web_learning_system.github_client is not None
        assert web_learning_system.headers['Authorization'] == 'token test-token'
        
        # Test clearing token
        web_learning_system.clear_github_token()
        assert web_learning_system.github_client is None
        assert 'Authorization' not in web_learning_system.headers

def test_github_token_clearing(web_learning_system):
    web_learning_system.set_github_token('test-token')
    web_learning_system.set_github_token('')  # Clear token
    assert web_learning_system.github_client is None
    assert 'Authorization' not in web_learning_system.headers 