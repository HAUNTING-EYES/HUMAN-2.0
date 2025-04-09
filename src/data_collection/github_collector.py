import os
import json
import logging
from typing import List, Dict, Any
from github import Github
from datetime import datetime
import time
from pathlib import Path

class GitHubDataCollector:
    def __init__(self, base_dir: str, github_token: str = None):
        self.base_dir = base_dir
        self.github_client = Github(github_token) if github_token else None
        self.logger = logging.getLogger(__name__)
        
        # Create data directories
        self.data_dir = Path(base_dir) / 'data'
        self.code_patterns_dir = self.data_dir / 'code_patterns'
        self.documentation_dir = self.data_dir / 'documentation'
        self.examples_dir = self.data_dir / 'examples'
        
        for directory in [self.data_dir, self.code_patterns_dir, 
                         self.documentation_dir, self.examples_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        
    def collect_from_repositories(self, repo_list: List[str], max_files: int = 100):
        """Collect data from specified GitHub repositories"""
        if not self.github_client:
            self.logger.error("GitHub client not initialized. Please provide a GitHub token.")
            return
            
        for repo_url in repo_list:
            try:
                self._rate_limit()
                owner, repo = self._parse_github_url(repo_url)
                repository = self.github_client.get_repo(f"{owner}/{repo}")
                
                # Collect repository data
                repo_data = {
                    'repo_url': repo_url,
                    'name': repository.name,
                    'description': repository.description,
                    'stars': repository.stargazers_count,
                    'language': repository.language,
                    'topics': repository.get_topics(),
                    'code_patterns': [],
                    'documentation': [],
                    'examples': []
                }
                
                # Process repository contents
                self._process_repository_contents(repository, repo_data, max_files)
                
                # Save collected data
                self._save_repository_data(repo_data)
                
            except Exception as e:
                self.logger.error(f"Error collecting data from {repo_url}: {str(e)}")
                
    def _process_repository_contents(self, repository, repo_data: Dict[str, Any], max_files: int):
        """Process repository contents and extract patterns"""
        try:
            contents = repository.get_contents("")
            if not isinstance(contents, list):
                contents = [contents]
                
            file_count = 0
            for content in contents:
                if file_count >= max_files:
                    break
                    
                if content.type == "file":
                    try:
                        self._rate_limit()
                        file_content = content.decoded_content.decode('utf-8')
                        file_type = content.name.split('.')[-1].lower()
                        
                        # Process based on file type
                        if file_type in ['py', 'js', 'java', 'cpp', 'cs']:
                            pattern = self._extract_code_pattern(content.name, file_content, file_type)
                            repo_data['code_patterns'].append(pattern)
                        elif file_type in ['md', 'txt', 'rst']:
                            doc = self._process_documentation(content.name, file_content)
                            repo_data['documentation'].append(doc)
                        elif file_type in ['json', 'yaml', 'yml']:
                            example = self._process_example(content.name, file_content)
                            repo_data['examples'].append(example)
                            
                        file_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing file {content.path}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error processing repository contents: {str(e)}")
            
    def _extract_code_pattern(self, filename: str, content: str, file_type: str) -> Dict[str, Any]:
        """Extract code pattern from file content"""
        return {
            'pattern_id': f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'language': file_type,
            'code': content,
            'description': f"Code pattern from {filename}",
            'tags': self._extract_tags(content),
            'metadata': {
                'complexity': self._calculate_complexity(content),
                'quality_score': self._calculate_quality_score(content),
                'usage_count': 0
            }
        }
        
    def _process_documentation(self, filename: str, content: str) -> Dict[str, Any]:
        """Process documentation content"""
        return {
            'doc_id': f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'content': content,
            'type': 'documentation',
            'metadata': {
                'source': filename,
                'language': 'en',
                'embedding': []  # Will be filled by NLP processing
            }
        }
        
    def _process_example(self, filename: str, content: str) -> Dict[str, Any]:
        """Process example content"""
        return {
            'example_id': f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'content': content,
            'type': 'example',
            'metadata': {
                'source': filename,
                'format': filename.split('.')[-1],
                'embedding': []  # Will be filled by NLP processing
            }
        }
        
    def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        # Simple tag extraction - can be enhanced with NLP
        tags = set()
        lines = content.split('\n')
        for line in lines:
            if '#' in line and 'TODO' in line:
                tags.add('todo')
            if 'FIXME' in line:
                tags.add('fixme')
            if 'def ' in line or 'class ' in line:
                tags.add('definition')
        return list(tags)
        
    def _calculate_complexity(self, content: str) -> float:
        """Calculate code complexity"""
        # Simple complexity calculation - can be enhanced
        lines = content.split('\n')
        complexity = 0
        for line in lines:
            if any(keyword in line for keyword in ['if', 'for', 'while', 'except']):
                complexity += 1
        return complexity
        
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate code quality score"""
        # Simple quality score - can be enhanced
        lines = content.split('\n')
        score = 100.0
        
        # Penalize long lines
        for line in lines:
            if len(line) > 80:
                score -= 1
                
        # Penalize empty lines
        empty_lines = sum(1 for line in lines if not line.strip())
        score -= empty_lines * 0.5
        
        return max(0, score)
        
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = max(0, self.min_request_interval - time_since_last_request)
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        
    def _parse_github_url(self, url: str) -> tuple:
        """Parse GitHub URL to get owner and repository name"""
        parts = url.strip('/').split('/')
        if len(parts) < 2:
            raise ValueError("Invalid GitHub URL")
        return parts[-2], parts[-1]
        
    def _save_repository_data(self, repo_data: Dict[str, Any]):
        """Save collected repository data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        repo_name = repo_data['name'].replace('/', '_')
        
        # Save code patterns
        if repo_data['code_patterns']:
            patterns_file = self.code_patterns_dir / f"{repo_name}_{timestamp}_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump(repo_data['code_patterns'], f, indent=2)
                
        # Save documentation
        if repo_data['documentation']:
            docs_file = self.documentation_dir / f"{repo_name}_{timestamp}_docs.json"
            with open(docs_file, 'w') as f:
                json.dump(repo_data['documentation'], f, indent=2)
                
        # Save examples
        if repo_data['examples']:
            examples_file = self.examples_dir / f"{repo_name}_{timestamp}_examples.json"
            with open(examples_file, 'w') as f:
                json.dump(repo_data['examples'], f, indent=2)
                
        # Save repository metadata
        metadata_file = self.data_dir / f"{repo_name}_{timestamp}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'repo_url': repo_data['repo_url'],
                'name': repo_data['name'],
                'description': repo_data['description'],
                'stars': repo_data['stars'],
                'language': repo_data['language'],
                'topics': repo_data['topics'],
                'collection_date': timestamp
            }, f, indent=2) 