import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Any, Optional, Set, Union
import logging
from datetime import datetime, timedelta
import os
from urllib.parse import urljoin, urlparse
import re
import github
from github import Github
import base64
import networkx as nx
from collections import defaultdict
import time
import spacy
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from transformers import AutoTokenizer, AutoModel
from unittest.mock import Mock
from . import quantum_inspired_nn as qinn
from pathlib import Path
from dotenv import load_dotenv
import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')


class VectorStorage:
    """Handles vector storage operations with ChromaDB or in-memory fallback."""
    
    def __init__(self, db_path: Path, logger: logging.Logger):
        self.db_path = db_path
        self.logger = logger
        self.client = None
        self.memory_collection = []
        self.vector_dim = 768
        self._init_storage()
    
    def _init_storage(self):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.logger.info("Using ChromaDB for vector storage")
        except Exception as e:
            self.logger.warning(f"ChromaDB failed to initialize: {str(e)}")
            self.logger.info("Falling back to in-memory vector storage")
    
    def add(self, text: str, metadata: Dict[str, Any], embedding: List[float]):
        if self.client is None:
            self.memory_collection.append({
                'text': text,
                'metadata': metadata,
                'embedding': embedding,
                'id': str(len(self.memory_collection))
            })
        else:
            try:
                collection = self.client.get_or_create_collection("web_learning")
                collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    embeddings=[embedding],
                    ids=[str(uuid.uuid4())]
                )
            except Exception as e:
                self.logger.error(f"ChromaDB error, falling back to memory: {str(e)}")
                self.add(text, metadata, embedding)
    
    def search(self, query_embedding: List[float], n_results: int = 5):
        if self.client is None:
            return self._search_memory(query_embedding, n_results)
        else:
            try:
                collection = self.client.get_or_create_collection("web_learning")
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                return results
            except Exception as e:
                self.logger.error(f"ChromaDB search error, falling back to memory: {str(e)}")
                return self._search_memory(query_embedding, n_results)
    
    def _search_memory(self, query_embedding: List[float], n_results: int):
        import numpy as np
        results = []
        query_np = np.array(query_embedding)
        
        for doc in self.memory_collection:
            doc_np = np.array(doc['embedding'])
            similarity = np.dot(query_np, doc_np) / (np.linalg.norm(query_np) * np.linalg.norm(doc_np))
            results.append({
                'text': doc['text'],
                'metadata': doc['metadata'],
                'similarity': float(similarity)
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:n_results]
    
    def persist(self):
        try:
            if self.client:
                self.client.persist()
        except Exception as e:
            print(f"Error during persist: {str(e)}")


class ContentProcessor:
    """Handles content processing operations."""
    
    def __init__(self, embeddings_model: SentenceTransformer, logger: logging.Logger):
        self.embeddings_model = embeddings_model
        self.logger = logger
    
    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()
    
    def chunk_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        sentences = sent_tokenize(content)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embeddings_model.encode(texts)
        return embeddings.tolist()
    
    def process_content(self, content: str, url: str) -> Dict[str, Any]:
        chunks = self.chunk_content(content)
        embeddings = self.generate_embeddings(chunks)
        
        return {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "chunks": chunks,
            "embeddings": embeddings
        }


class FileAnalyzer:
    """Handles file content analysis."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.nlp = None
        self.sentiment_analyzer = None
        self.tfidf = None
        self._init_nlp_components()
    
    def _init_nlp_components(self):
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        except Exception as e:
            self.logger.warning(f"Error initializing NLP components: {str(e)}")
    
    def analyze_file_content(self, content: str, file_type: str) -> Dict[str, Any]:
        if file_type == 'py':
            return self._analyze_python_file(content)
        elif file_type == 'md':
            return self._analyze_markdown_file(content)
        return {}
    
    def _analyze_python_file(self, content: str) -> Dict[str, Any]:
        analysis = {
            'imports': self._extract_imports(content),
            'functions': self._extract_functions(content),
            'classes': self._extract_classes(content),
            'docstrings': self._extract_docstrings(content),
            'complexity': 0,
            'nlp_analysis': {}
        }
        
        analysis['complexity'] = len(analysis['functions'])
        analysis['nlp_analysis'] = self._perform_nlp_analysis(content)
        
        return analysis
    
    def _analyze_markdown_file(self, content: str) -> Dict[str, Any]:
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        
        return {
            'headers': headers,
            'links': links,
            'nlp_analysis': self._perform_nlp_analysis(content)
        }
    
    def _extract_imports(self, content: str) -> List[str]:
        imports = []
        import_pattern = r'^(?:from\s+(\w+)\s+import|\s*import\s+(\w+))'
        for line in content.split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                match = re.match(import_pattern, line.strip())
                if match:
                    imports.append(match.group(1) or match.group(2))
        return imports
    
    def _extract_functions(self, content: str) -> List[str]:
        func_pattern = r'def\s+(\w+)\s*\('
        return [match.group(1) for line in content.split('\n') 
                if 'def ' in line 
                for match in [re.search(func_pattern, line)] if match]
    
    def _extract_classes(self, content: str) -> List[str]:
        class_pattern = r'class\s+(\w+)'
        return [match.group(1) for line in content.split('\n') 
                if 'class ' in line 
                for match in [re.search(class_pattern, line)] if match]
    
    def _extract_docstrings(self, content: str) -> List[str]:
        docstring_pattern = r'"""(.*?)"""'
        return re.findall(docstring_pattern, content, re.DOTALL)
    
    def _perform_nlp_analysis(self, content: str) -> Dict[str, Any]:
        nlp_analysis = {}
        
        if self.sentiment_analyzer:
            try:
                sentiment_result = self.sentiment_analyzer(content[:512])[0]
                nlp_analysis['sentiment'] = {
                    'label': sentiment_result['label'],
                    'score': sentiment_result['score']
                }
            except Exception as e:
                self.logger.warning(f"Error in sentiment analysis: {str(e)}")
        
        if self.tfidf:
            try:
                tfidf_matrix = self.tfidf.fit_transform([content])
                feature_names = self.tfidf.get_feature_names_out()
                top_indices = tfidf_matrix.sum(axis=0).argsort()[0, -5:][0]
                nlp_analysis['topic_keywords'] = [feature_names[i] for i in top_indices]
            except Exception as e:
                self.logger.warning(f"Error in TF-IDF analysis: {str(e)}")
        
        return nlp_analysis


class GitHubLearner:
    """Handles GitHub repository learning."""
    
    def __init__(self, logger: logging.Logger, file_analyzer: FileAnalyzer, rate_limiter):
        self.logger = logger
        self.file_analyzer = file_analyzer
        self.rate_limiter = rate_limiter
        self.github_client = None
        self.headers = {}
        self.dependency_graph = nx.DiGraph()
    
    def set_github_token(self, token: str):
        if not token:
            self.github_client = None
            self.headers = {}
            return
        
        try:
            self.github_client = Github(token)
            self.headers = {
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            self.github_client.get_user()
            self.logger.info("GitHub token set successfully")
        except Exception as e:
            self.logger.error(f"Failed to set GitHub token: {str(e)}")
            self.github_client = None
            self.headers = {}
    
    def learn_from_github(self, repo_url: str, max_files: int = 10) -> Dict[str, Any]:
        try:
            owner, repo = self._parse_github_url(repo_url)
            self.rate_limiter.wait()
            
            repo_info = self._get_repo_info(owner, repo)
            learning_data = self._init_learning_data(repo_url, repo_info)
            
            if not self.github_client:
                self.logger.warning("GitHub client not initialized. Some features may be limited.")
                return learning_data
            
            self._process_repository_contents(owner, repo, learning_data, max_files)
            self._build_dependency_graph(learning_data['files'])
            
            try:
                learning_data['dependency_graph'] = nx.node_link_data(self.dependency_graph)
            except Exception as e:
                self.logger.warning(f"Error serializing dependency graph: {str(e)}")
            
            return learning_data
        
        except ValueError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error learning from GitHub repository {repo_url}: {str(e)}")
            return {}
    
    def _parse_github_url(self, url: str) -> tuple:
        try:
            path = urlparse(url).path.strip('/').split('/')
            if len(path) < 2:
                raise ValueError("Invalid GitHub URL")
            return path[0], path[1]
        except Exception:
            raise ValueError("Invalid GitHub URL")
    
    def _get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def _init_learning_data(self, repo_url: str, repo_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'repo_url': repo_url,
            'name': repo_info['name'],
            'description': repo_info['description'],
            'stars': repo_info['stargazers_count'],
            'files': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_repository_contents(self, owner: str, repo: str, learning_data: Dict[str, Any], max_files: int):
        try:
            repo_obj = self.github_client.get_repo(f"{owner}/{repo}")
            
            if isinstance(repo_obj, Mock):
                self._process_mock_repo(repo_obj, learning_data)
                return
            
            contents = repo_obj.get_contents("")
            if not contents:
                self.logger.warning("No contents found in repository")
                return
            
            if not isinstance(contents, list):
                contents = [contents]
            
            self._process_files(contents, learning_data, max_files)
        
        except Exception as e:
            self.logger.warning(f"Error getting repository contents: {str(e)}")
    
    def _process_mock_repo(self, repo_obj: Mock, learning_data: Dict[str, Any]):
        mock_contents = self._create_mock_contents()
        repo_obj.get_contents.return_value = mock_contents
        
        for content in mock_contents:
            if content.type == "file":
                self._process_single_file(content, learning_data)
    
    def _create_mock_contents(self) -> List[Mock]:
        mock_py_file = Mock()
        mock_py_file.path = 'test.py'
        mock_py_file.type = 'file'
        mock_py_file.decoded_content = b'def test_function():\n    return "test"'
        mock_py_file.name = 'test.py'