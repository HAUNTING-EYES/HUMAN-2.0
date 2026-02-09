import pytest
import torch
from src.embedder.code_embedder import CodeEmbedder, CodeEmbedding

@pytest.fixture
def embedder():
    return CodeEmbedder(device='cpu')

@pytest.fixture
def sample_code():
    return '''
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
'''

def test_embedder_initialization(embedder):
    """Test embedder initialization."""
    assert embedder.device == 'cpu'
    assert embedder.max_length == 512
    assert embedder.embedding_dim == 768
    assert embedder.tokenizer is not None
    assert embedder.base_model is not None

def test_preprocess_code(embedder, sample_code):
    """Test code preprocessing."""
    processed_code = embedder.preprocess_code(sample_code)
    assert '<id>total</id>' in processed_code
    assert '<id>items</id>' in processed_code
    assert '<id>item</id>' in processed_code

def test_tokenize(embedder, sample_code):
    """Test code tokenization."""
    tokens = embedder.tokenize(sample_code)
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens
    assert isinstance(tokens['input_ids'], torch.Tensor)
    assert isinstance(tokens['attention_mask'], torch.Tensor)

def test_forward(embedder, sample_code):
    """Test forward pass."""
    embedding = embedder.forward(sample_code)
    assert isinstance(embedding, CodeEmbedding)
    assert isinstance(embedding.embedding, torch.Tensor)
    assert embedding.embedding.shape == (1, 768)
    assert len(embedding.tokens) > 0
    assert isinstance(embedding.attention_mask, torch.Tensor)

def test_embed_batch(embedder):
    """Test batch embedding."""
    code_snippets = [
        'def func1(): return 1',
        'def func2(): return 2'
    ]
    embeddings = embedder.embed_batch(code_snippets)
    assert len(embeddings) == 2
    assert all(isinstance(emb, CodeEmbedding) for emb in embeddings)

def test_similarity(embedder):
    """Test similarity calculation."""
    code1 = 'def func1(): return 1'
    code2 = 'def func2(): return 2'
    emb1 = embedder.forward(code1).embedding
    emb2 = embedder.forward(code2).embedding
    similarity = embedder.similarity(emb1, emb2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

def test_find_similar_snippets(embedder):
    """Test finding similar code snippets."""
    query = 'def calculate(x): return x + 1'
    snippets = [
        'def add(x): return x + 1',
        'def multiply(x): return x * 2',
        'def subtract(x): return x - 1'
    ]
    similar = embedder.find_similar_snippets(query, snippets, threshold=0.5)
    assert len(similar) > 0
    assert all(isinstance(s, tuple) and len(s) == 2 for s in similar)
    assert all(isinstance(s[1], float) and 0.5 <= s[1] <= 1.0 for s in similar) 