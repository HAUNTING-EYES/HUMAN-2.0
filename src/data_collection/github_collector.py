import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from github import Github
from github.Repository import Repository
from github.ContentFile import ContentFile
from datetime import datetime
import asyncio
import aiofiles
from pathlib import Path

class GitHubDataCollector:
    SUPPORTED_CODE_EXTENSIONS = {'py', 'js', 'java', 'cpp', 'cs'}
    SUPPORTED_DOC_EXTENSIONS = {'md', 'txt', 'rst'}
    SUPPORTED_EXAMPLE_EXTENSIONS = {'json', 'yaml', 'yml'}
    
    COMPLEXITY_KEYWORDS = ['if', 'for', 'while', 'except']
    MAX_LINE_LENGTH = 80
    TAG_KEYWORDS = {
        'todo': ['TODO'],
        'fixme': ['FIXME'],
        'definition': ['def ', 'class ']
    }
    
    def __init__(self, base_dir: str, github_token: str = None):
        self.base_dir = base_dir
        self.github_client = Github(github_token) if github_token else None
        self.logger = logging.getLogger(__name__)
        self._setup_directories()
        
    def _setup_directories(self):
        self.data_dir = Path(self.base_dir) / 'data'
        self.code_patterns_dir = self.data_dir / 'code_patterns'
        self.documentation_dir = self.data_dir / 'documentation'
        self.examples_dir = self.data_dir / 'examples'
        
        directories = [self.data_dir, self.code_patterns_dir, 
                      self.documentation_dir, self.examples_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    async def collect_from_repositories_async(self, repo_list: List[str], max_files: int = 100):
        if not self._validate_github_client():
            return
            
        tasks = [self._collect_from_single_repository_async(repo_url, max_files) 
                 for repo_url in repo_list]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def collect_from_repositories(self, repo_list: List[str], max_files: int = 100):
        if not self._validate_github_client():
            return
            
        for repo_url in repo_list:
            self._collect_from_single_repository(repo_url, max_files)
                
    def _validate_github_client(self) -> bool:
        if not self.github_client:
            self.logger.error("GitHub client not initialized. Please provide a GitHub token.")
            return False
        return True
    
    async def _collect_from_single_repository_async(self, repo_url: str, max_files: int):
        try:
            repository = self._get_repository(repo_url)
            repo_data = self._create_repo_data_structure(repository, repo_url)
            await self._process_repository_contents_async(repository, repo_data, max_files)
            await self._save_repository_data_async(repo_data)
        except Exception as e:
            self.logger.error(f"Error collecting data from {repo_url}: {str(e)}")
    
    def _collect_from_single_repository(self, repo_url: str, max_files: int):
        try:
            repository = self._get_repository(repo_url)
            repo_data = self._create_repo_data_structure(repository, repo_url)
            self._process_repository_contents(repository, repo_data, max_files)
            self._save_repository_data(repo_data)
        except Exception as e:
            self.logger.error(f"Error collecting data from {repo_url}: {str(e)}")
    
    def _get_repository(self, repo_url: str) -> Repository:
        owner, repo = self._parse_github_url(repo_url)
        return self.github_client.get_repo(f"{owner}/{repo}")
    
    def _create_repo_data_structure(self, repository: Repository, repo_url: str) -> Dict[str, Any]:
        return {
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
    
    async def _process_repository_contents_async(self, repository: Repository, repo_data: Dict[str, Any], max_files: int):
        try:
            contents = self._get_repository_contents(repository)
            await self._process_files_async(contents, repo_data, max_files)
        except Exception as e:
            self.logger.error(f"Error processing repository contents: {str(e)}")
                
    def _process_repository_contents(self, repository: Repository, repo_data: Dict[str, Any], max_files: int):
        try:
            contents = self._get_repository_contents(repository)
            self._process_files(contents, repo_data, max_files)
        except Exception as e:
            self.logger.error(f"Error processing repository contents: {str(e)}")
    
    def _get_repository_contents(self, repository: Repository) -> List[ContentFile]:
        contents = repository.get_contents("")
        return contents if isinstance(contents, list) else [contents]
    
    async def _process_files_async(self, contents: List[ContentFile], repo_data: Dict[str, Any], max_files: int):
        file_count = 0
        for content in contents:
            if file_count >= max_files:
                break
            
            if await self._process_single_file_async(content, repo_data):
                file_count += 1
    
    def _process_files(self, contents: List[ContentFile], repo_data: Dict[str, Any], max_files: int):
        file_count = 0
        for content in contents:
            if file_count >= max_files:
                break
            
            if self._process_single_file(content, repo_data):
                file_count += 1
    
    async def _process_single_file_async(self, content: ContentFile, repo_data: Dict[str, Any]) -> bool:
        try:
            file_content = content.decoded_content.decode('utf-8')
            file_type = self._get_file_extension(content.name)
            self._categorize_and_store_file(content.name, file_content, file_type, repo_data)
            return True
        except Exception as e:
            self.logger.warning(f"Error processing file {content.path}: {str(e)}")
            return False
    
    def _process_single_file(self, content: ContentFile, repo_data: Dict[str, Any]) -> bool:
        try:
            file_content = content.decoded_content.decode('utf-8')
            file_type = self._get_file_extension(content.name)
            self._categorize_and_store_file(content.name, file_content, file_type, repo_data)
            return True
        except Exception as e:
            self.logger.warning(f"Error processing file {content.path}: {str(e)}")
            return False
    
    def _get_file_extension(self, filename: str) -> str:
        return filename.split('.')[-1].lower()
    
    def _categorize_and_store_file(self, filename: str, content: str, file_type: str, repo_data: Dict[str, Any]):
        category_handlers = {
            'code_patterns': (self.SUPPORTED_CODE_EXTENSIONS, self._extract_code_pattern),
            'documentation': (self.SUPPORTED_DOC_EXTENSIONS, self._process_documentation),
            'examples': (self.SUPPORTED_EXAMPLE_EXTENSIONS, self._process_example)
        }
        
        for category, (extensions, handler) in category_handlers.items():
            if file_type in extensions:
                if category == 'code_patterns':
                    processed_data = handler(filename, content, file_type)
                else:
                    processed_data = handler(filename, content)
                repo_data[category].append(processed_data)
                break
            
    def _extract_code_pattern(self, filename: str, content: str, file_type: str) -> Dict[str, Any]:
        return {
            'pattern_id': self._generate_id(filename),
            'language': file_type,
            'code': content,
            'description': f"Code pattern from {filename}",
            'tags': self._extract_tags(content),
            'metadata': self._create_code_metadata(content)
        }
    
    def _create_code_metadata(self, content: str) -> Dict[str, Any]:
        return {
            'complexity': self._calculate_complexity(content),
            'quality_score': self._calculate_quality_score(content),
            'usage_count': 0
        }
        
    def _process_documentation(self, filename: str, content: str) -> Dict[str, Any]:
        return {
            'doc_id': self._generate_id(filename),
            'content': content,
            'type': 'documentation',
            'metadata': {
                'source': filename,
                'language': 'en',
                'embedding': []
            }
        }
        
    def _process_example(self, filename: str, content: str) -> Dict[str, Any]:
        return {
            'example_id': self._generate_id(filename),
            'content': content,
            'type': 'example',
            'metadata': {
                'source': filename,
                'format': self._get_file_extension(filename),
                'embedding': []
            }
        }
    
    def _generate_id(self, filename: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{filename}_{timestamp}"
        
    def _extract_tags(self, content: str) -> List[str]:
        tags = set()
        lines = content.split('\n')
        for line in lines:
            for tag, keywords in self.TAG_KEYWORDS.items():
                if any(keyword in line for keyword in keywords):
                    tags.add(tag)
        return list(tags)
        
    def _calculate_complexity(self, content: str) -> int:
        lines = content.split('\n')
        return sum(1 for line in lines 
                   if any(keyword in line for keyword in self.COMPLEXITY_KEYWORDS))
        
    def _calculate_quality_score(self, content: str) -> float:
        lines = content.split('\n')
        score = 100.0
        
        long_lines = sum(1 for line in lines if len(line) > self.MAX_LINE_LENGTH)
        empty_lines = sum(1 for line in lines if not line.strip())
        
        score -= long_lines
        score -= empty_lines * 0.5
        
        return max(0, score)
        
    def _parse_github_url(self, url: str) -> Tuple[str, str]:
        parts = url.strip('/').split('/')
        if len(parts) < 2:
            raise ValueError("Invalid GitHub URL")
        return parts[-2], parts[-1]
    
    async def _save_repository_data_async(self, repo_data: Dict[str, Any]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        repo_name = repo_data['name'].replace('/', '_')
        
        tasks = self._create_save_tasks_async(repo_data, repo_name, timestamp)
        await asyncio.gather(*tasks)
        
    def _create_save_tasks_async(self, repo_data: Dict[str, Any], repo_name: str, timestamp: str) -> List:
        save_operations = [
            (repo_data['code_patterns'], self.code_patterns_dir, 'patterns'),
            (repo_data['documentation'], self.documentation_dir, 'docs'),
            (repo_data['examples'], self.examples_dir, 'examples')
        ]
        
        tasks = []
        for data, directory, suffix in save_operations:
            if data:
                filepath = directory / f"{repo_name}_{timestamp}_{suffix}.json"
                tasks.append(self._write_json_file_async(filepath, data))
        
        tasks.append(self._save_metadata_async(repo_data, repo_name, timestamp))
        return tasks
        
    def _save_repository_data(self, repo_data: Dict[str, Any]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        repo_name = repo_data['name'].replace('/', '_')
        
        self._execute_save_operations(repo_data, repo_name, timestamp)
        self._save_metadata(repo_data, repo_name, timestamp)
    
    def _execute_save_operations(self, repo_data: Dict[str, Any], repo_name: str, timestamp: str):
        save_operations = [
            (repo_data['code_patterns'], self.code_patterns_dir, 'patterns'),
            (repo_data['documentation'], self.documentation_dir, 'docs'),
            (repo_data['examples'], self.examples_dir, 'examples')
        ]
        
        for data, directory, suffix in save_operations:
            if data:
                filepath = directory / f"{repo_name}_{timestamp}_{suffix}.json"
                self._write_json_file(filepath, data)
    
    async def _save_metadata_async(self, repo_data: Dict[str, Any], repo_name: str, timestamp: str):
        metadata_file = self.data_dir / f"{repo_name}_{timestamp}_metadata.json"
        metadata = self._build_metadata(repo_data, timestamp)
        await self._write_json_file_async(metadata_file, metadata)
    
    def _save_metadata(self, repo_data: Dict[str, Any], repo_name: str, timestamp: str):
        metadata_file = self.data_dir / f"{repo_name}_{timestamp}_metadata.json"
        metadata = self._build_metadata(repo_data, timestamp)
        self._write_json_file(metadata_file, metadata)
    
    def _build_metadata(self, repo_data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        return {
            'repo_url': repo_data['repo_url'],
            'name': repo_data['name'],
            'description': repo_data['description'],
            'stars': repo_data['stars'],
            'language': repo_data['language'],
            'topics': repo_data['topics'],
            'collection_date': timestamp
        }
    
    async def _write_json_file_async(self, filepath: Path, data: Any):
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(data, indent=2))
    
    def _write_json_file(self, filepath: Path, data: Any):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)