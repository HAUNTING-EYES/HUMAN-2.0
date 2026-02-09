import pytest
import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent)
sys.path.insert(0, src_path)

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables and paths."""
    os.environ['TESTING'] = 'true'
    yield
    if 'TESTING' in os.environ:
        del os.environ['TESTING'] 