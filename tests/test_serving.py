import pytest
from fastapi.testclient import TestClient
from src.serving.app import app, ModelServer
import torch

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def model_server(tmp_path):
    # Create a dummy model for testing
    model_path = tmp_path / "test_model.zip"
    torch.save({}, model_path)
    return ModelServer(str(model_path), device='cpu')

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_optimize_code(client, monkeypatch):
    """Test code optimization endpoint."""
    # Mock the optimize_code method
    def mock_optimize(*args, **kwargs):
        return {
            "original_code": "def test(): pass",
            "optimized_code": "def test_optimized(): pass",
            "metrics": {"complexity": 1.0},
            "steps_taken": 1,
            "improvements": ["Step 1: Improved naming"]
        }
    
    monkeypatch.setattr("src.serving.app.model_server.optimize_code", mock_optimize)
    
    response = client.post(
        "/optimize",
        json={
            "code": "def test(): pass",
            "max_steps": 5,
            "optimization_type": "all"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "optimized_code" in data
    assert "metrics" in data
    assert "steps_taken" in data
    assert "improvements" in data

def test_model_server_initialization(model_server):
    """Test model server initialization."""
    assert model_server.device == 'cpu'
    assert model_server.model is not None
    assert model_server.code_embedder is not None

def test_model_server_optimize_code(model_server):
    """Test model server code optimization."""
    code = '''
def test_function():
    x = 1
    return x
'''
    result = model_server._optimize_code(code)
    assert result.original_code == code
    assert isinstance(result.optimized_code, str)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.steps_taken, int)
    assert isinstance(result.improvements, list)

def test_invalid_code(client, monkeypatch):
    """Test handling of invalid code."""
    response = client.post(
        "/optimize",
        json={
            "code": "invalid python code",
            "max_steps": 5
        }
    )
    assert response.status_code == 500

def test_missing_code(client):
    """Test handling of missing code."""
    response = client.post(
        "/optimize",
        json={
            "max_steps": 5
        }
    )
    assert response.status_code == 422

def test_invalid_max_steps(client):
    """Test handling of invalid max_steps."""
    response = client.post(
        "/optimize",
        json={
            "code": "def test(): pass",
            "max_steps": -1
        }
    )
    assert response.status_code == 422

def test_caching(model_server):
    """Test response caching."""
    code = "def test(): return 1"
    
    # First call
    result1 = model_server.optimize_code(code)
    
    # Second call (should be cached)
    result2 = model_server.optimize_code(code)
    
    assert result1 == result2 