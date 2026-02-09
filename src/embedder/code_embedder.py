from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import ast
from dataclasses import dataclass

@dataclass
class CodeEmbedding:
    """Represents an embedded code snippet."""
    embedding: torch.Tensor
    tokens: List[str]
    attention_mask: torch.Tensor

class CodeEmbedder(nn.Module):
    """Transformer-based code embedding model."""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        embedding_dim: int = 768,
        device: Optional[str] = None
    ):
        super().__init__()
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and base model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.base_model = RobertaModel.from_pretrained(model_name)
        
        # Configuration
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Additional layers for code-specific features
        self.code_feature_layer = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Move model to device
        self.to(self.device)
        
    def preprocess_code(self, code: str) -> str:
        """Preprocess code for better tokenization."""
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Extract identifiers and string literals
            identifiers = []
            string_literals = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    identifiers.append(node.id)
                elif isinstance(node, ast.Str):
                    string_literals.append(node.s)
            
            # Add special tokens for code structure
            processed_code = code
            for identifier in set(identifiers):
                processed_code = processed_code.replace(
                    identifier,
                    f"<id>{identifier}</id>"
                )
            for literal in set(string_literals):
                processed_code = processed_code.replace(
                    f'"{literal}"',
                    f'<str>"{literal}"</str>'
                )
            
            return processed_code
            
        except SyntaxError:
            # If code can't be parsed, return as is
            return code
    
    def tokenize(self, code: str) -> Dict[str, torch.Tensor]:
        """Tokenize code snippet."""
        # Preprocess code
        processed_code = self.preprocess_code(code)
        
        # Tokenize
        tokens = self.tokenizer(
            processed_code,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        return {
            key: val.to(self.device)
            for key, val in tokens.items()
        }
    
    def forward(self, code: str) -> CodeEmbedding:
        """Generate embedding for code snippet."""
        # Tokenize code
        tokens = self.tokenize(code)
        
        # Get base embeddings
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask']
            )
        
        # Get sequence embedding (CLS token)
        sequence_embedding = base_outputs.last_hidden_state[:, 0, :]
        
        # Apply code-specific features
        code_embedding = self.code_feature_layer(sequence_embedding)
        
        return CodeEmbedding(
            embedding=code_embedding,
            tokens=self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]),
            attention_mask=tokens['attention_mask']
        )
    
    def embed_batch(self, code_snippets: List[str]) -> List[CodeEmbedding]:
        """Generate embeddings for multiple code snippets."""
        return [self.forward(code) for code in code_snippets]
    
    def similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings."""
        return torch.nn.functional.cosine_similarity(
            embedding1,
            embedding2,
            dim=-1
        ).item()
    
    def find_similar_snippets(
        self,
        query_code: str,
        code_snippets: List[str],
        threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """Find similar code snippets using embeddings."""
        # Get query embedding
        query_embedding = self.forward(query_code).embedding
        
        # Get embeddings for all snippets
        snippet_embeddings = [
            self.forward(snippet).embedding
            for snippet in code_snippets
        ]
        
        # Calculate similarities
        similarities = [
            self.similarity(query_embedding, snippet_embedding)
            for snippet_embedding in snippet_embeddings
        ]
        
        # Filter by threshold and sort by similarity
        similar_snippets = [
            (snippet, similarity)
            for snippet, similarity in zip(code_snippets, similarities)
            if similarity >= threshold
        ]
        
        return sorted(similar_snippets, key=lambda x: x[1], reverse=True) 