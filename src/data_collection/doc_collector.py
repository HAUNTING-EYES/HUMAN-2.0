import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import markdown
from datetime import datetime
import time
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class DocumentationCollector:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        
        # Create documentation directory
        self.docs_dir = Path(base_dir) / 'data' / 'documentation'
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP components
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info("NLP components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {str(e)}")
            self.tokenizer = None
            self.model = None
            
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        
    def collect_from_urls(self, url_list: List[str]):
        """Collect documentation from specified URLs"""
        for url in url_list:
            try:
                self._rate_limit()
                content = self._fetch_url_content(url)
                if content:
                    doc_data = self._process_content(url, content)
                    self._save_documentation(doc_data)
                    
            except Exception as e:
                self.logger.error(f"Error collecting from {url}: {str(e)}")
                
    def collect_from_files(self, file_list: List[str]):
        """Collect documentation from local files"""
        for file_path in file_list:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                doc_data = self._process_content(file_path, content)
                self._save_documentation(doc_data)
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
                
    def _fetch_url_content(self, url: str) -> str:
        """Fetch content from URL"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    def _process_content(self, source: str, content: str) -> Dict[str, Any]:
        """Process documentation content"""
        # Convert markdown to text if needed
        if source.endswith('.md'):
            content = markdown.markdown(content)
            
        # Generate embeddings if NLP components are available
        embedding = []
        if self.tokenizer and self.model:
            embedding = self._generate_embedding(content)
            
        return {
            'doc_id': f"{Path(source).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'source': source,
            'content': content,
            'type': 'documentation',
            'metadata': {
                'language': 'en',
                'embedding': embedding,
                'collection_date': datetime.now().isoformat()
            }
        }
        
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using transformer model"""
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Convert to list and normalize
            embedding = embeddings[0].numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return []
            
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = max(0, self.min_request_interval - time_since_last_request)
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        
    def _save_documentation(self, doc_data: Dict[str, Any]):
        """Save processed documentation"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_name = Path(doc_data['source']).stem.replace('/', '_')
        
        # Save documentation
        doc_file = self.docs_dir / f"{source_name}_{timestamp}_doc.json"
        with open(doc_file, 'w') as f:
            json.dump(doc_data, f, indent=2)
            
        self.logger.info(f"Saved documentation from {doc_data['source']}") 