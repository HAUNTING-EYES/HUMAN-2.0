#!/usr/bin/env python3
"""
HUMAN 2.0 Autonomous Learner V2
Enhanced autonomous learning with GitHub integration.

Major improvements over V1:
1. GitHub learning connected to curiosity engine
2. Pattern extraction from GitHub repos
3. Storage in ChromaDB for use by self-improvement
4. Feedback loop to curiosity engine (reinforcement learning)
5. Multi-source learning (web + GitHub)
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import aiohttp

sys.path.append(str(Path(__file__).parent.parent))

from core.autonomous_learner_v1 import (
    AutonomousLearnerV1,
    WebSearcher,
    KnowledgeExtractor,
    KnowledgeImplementer
)
from core.code_embedder import CodeEmbedder
from components.external_learning import ExternalLearning
from components.github_integration import GitHubIntegration
from consciousness.curiosity import CuriosityEngine, Question, KnowledgeDomain

load_dotenv()


class LLMClient:
    """Shared LLM client for API calls."""
    
    BASE_URL = 'https://api.together.xyz/v1/chat/completions'
    DEFAULT_TEMPERATURE = 0.5
    DEFAULT_MAX_TOKENS = 100
    REQUEST_TIMEOUT = 30
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    async def call_async(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Async LLM call."""
        temperature = temperature or self.DEFAULT_TEMPERATURE
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        
        try:
            headers = self._build_headers()
            data = self._build_request_data(prompt, temperature, max_tokens)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return self._extract_content(result)
        
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _build_request_data(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Build request data."""
        return {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
            'max_tokens': max_tokens
        }
    
    def _extract_content(self, result: Dict[str, Any]) -> str:
        """Extract and clean content from API response."""
        content = result['choices'][0]['message']['content'].strip()
        return content.strip('"').strip("'")


class GitHubSearcher:
    """Convert curiosity questions to GitHub searches."""

    PROMPT_TEMPLATE = """Convert this curiosity question to a GitHub repository search query.

Question: {question}
Domain: {domain}

Return a search query that will find relevant GitHub repositories implementing or discussing this topic.
Focus on code repositories, not documentation.

Examples:
- Question: "How can neural networks learn incrementally without forgetting?"
  Search: "continual learning neural network python"

- Question: "What are efficient ways to implement graph neural networks?"
  Search: "graph neural network pytorch implementation"

- Question: "How to implement meta-learning for few-shot classification?"
  Search: "meta-learning few-shot classification python"

Return ONLY the search query, no explanation:"""

    def __init__(self, llm_client: LLMClient):
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client

    async def question_to_search_terms_async(self, question: Question) -> str:
        """Convert curiosity question to GitHub search query."""
        prompt = self.PROMPT_TEMPLATE.format(
            question=question.content,
            domain=question.domain.value
        )
        
        try:
            return await self.llm_client.call_async(prompt)
        except Exception as e:
            self.logger.error(f"Failed to convert question to search terms: {e}")
            return question.content


class GitHubRepoSearcher:
    """Search GitHub repositories."""
    
    BASE_URL = "https://api.github.com/search/repositories"
    MIN_STARS = 100
    REQUEST_TIMEOUT = 10
    
    def __init__(self, github_token: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.github_token = github_token
    
    async def search_repos_async(self, query: str, max_repos: int = 3) -> List[str]:
        """Search for GitHub repositories."""
        try:
            headers = self._build_headers()
            params = self._build_search_params(query, max_repos)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return self._extract_repo_urls(data)

        except Exception as e:
            self.logger.error(f"GitHub search failed: {e}")
            return []
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        return headers
    
    def _build_search_params(self, query: str, max_repos: int) -> Dict[str, Any]:
        """Build search parameters."""
        return {
            'q': f'{query} language:python stars:>{self.MIN_STARS}',
            'sort': 'stars',
            'order': 'desc',
            'per_page': max_repos
        }
    
    def _extract_repo_urls(self, data: Dict[str, Any]) -> List[str]:
        """Extract repository URLs from API response."""
        return [item['html_url'] for item in data.get('items', [])]


class KnowledgeMerger:
    """Merge knowledge from multiple sources."""
    
    @staticmethod
    def merge(web_knowledge: Optional[Dict[str, Any]], 
              github_knowledge: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge knowledge from web and GitHub sources."""
        merged = KnowledgeMerger._create_empty_knowledge()
        
        sources = [
            (web_knowledge, 'web'),
            (github_knowledge, 'github')
        ]
        
        for knowledge, source_name in sources:
            if knowledge:
                KnowledgeMerger._merge_single_source(merged, knowledge, source_name)

        KnowledgeMerger._finalize_merge(merged)
        return merged
    
    @staticmethod
    def _create_empty_knowledge() -> Dict[str, Any]:
        """Create empty knowledge structure."""
        return {
            'summary': '',
            'key_concepts': [],
            'code_examples': [],
            'related_topics': [],
            'implementation_ideas': [],
            'confidence': 0.0,
            'sources': []
        }
    
    @staticmethod
    def _merge_single_source(merged: Dict[str, Any], knowledge: Dict[str, Any], source_name: str):
        """Merge a single knowledge source."""
        merged['summary'] += knowledge.get('summary', '') + ' '
        merged['key_concepts'].extend(knowledge.get('key_concepts', []))
        merged['code_examples'].extend(knowledge.get('code_examples', []))
        merged['related_topics'].extend(knowledge.get('related_topics', []))
        merged['implementation_ideas'].extend(knowledge.get('implementation_ideas', []))
        merged['confidence'] += knowledge.get('confidence', 0.0)
        merged['sources'].append(source_name)
    
    @staticmethod
    def _finalize_merge(merged: Dict[str, Any]):
        """Finalize merged knowledge."""
        if merged['sources']:
            merged['confidence'] /= len(merged['sources'])
        
        merged['key_concepts'] = list(set(merged['key_concepts']))
        merged['related_topics'] = list(set(merged['related_topics']))


class FeedbackProvider:
    """Provide feedback to curiosity engine."""
    
    REWARD_THRESHOLDS = [
        (0.7, 1.0),
        (0.5, 0.5),
        (0.3, 0.0),
        (0.0, -0.3)
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feedback_history = []
    
    def calculate_reward(self, confidence: float) -> float:
        """Calculate reward based on confidence."""
        for threshold, reward in self.REWARD_THRESHOLDS:
            if confidence > threshold:
                return reward
        return -0.3
    
    async def provide_feedback_async(self, question: Question, knowledge: Dict[str, Any], 
                                    curiosity_engine: CuriosityEngine):
        """Provide feedback to curiosity engine."""
        confidence = knowledge.get('confidence', 0.0)
        reward = self.calculate_reward(confidence)

        await self._send_feedback_to_engine_async(curiosity_engine, question, reward)
        self._record_feedback(question, reward, confidence)
    
    async def _send_feedback_to_engine_async(self, curiosity_engine: CuriosityEngine, 
                                            question: Question, reward: float):
        """Send feedback to curiosity engine."""
        try:
            if hasattr(curiosity_engine, 'reinforce_question'):
                if asyncio.iscoroutinefunction(curiosity_engine.reinforce_question):
                    await curiosity_engine.reinforce_question(question.id, reward)
                else:
                    curiosity_engine.reinforce_question(question.id, reward)
                self.logger.info(f"Provided feedback for question: reward={reward:.2f}")
            else:
                self.logger.warning("CuriosityEngine missing reinforce_question() method")
        except Exception as e:
            self.logger.error(f"Error providing feedback: {e}")
    
    def _record_feedback(self, question: Question, reward: float, confidence: float):
        """Record feedback in history."""
        self.feedback_history.append({
            'question': question.content,
            'question_id': question.id,
            'reward': reward,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })


class ComponentInitializer:
    """Initialize all components for AutonomousLearnerV2."""
    
    DEFAULT_MODEL = 'meta-llama/Meta-Llama-3-70B-Instruct'
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    async def initialize_all_async(self) -> Dict[str, Any]:
        """Initialize all components."""
        self.logger.info("Initializing GitHub learning components...")
        
        llm_client = self._init_llm_client()
        
        components = {
            'llm_client': llm_client,
            'code_embedder': CodeEmbedder(),
            'knowledge_merger': KnowledgeMerger(),
            'feedback_provider': FeedbackProvider(),
            'external_learning': await self._init_external_learning_async(),
            'github_integration': await self._init_github_integration_async(),
            'github_searcher': GitHubSearcher(llm_client),
            'repo_searcher': GitHubRepoSearcher(os.getenv('GITHUB_TOKEN'))
        }
        
        self.logger.info("All components initialized successfully")
        return components
    
    def _init_llm_client(self) -> LLMClient:
        """Initialize LLM client."""
        together_api_key = os.getenv('TOGETHER_API_KEY')
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment")
        
        return LLMClient(
            api_key=together_api_key,
            model=self.DEFAULT_MODEL
        )
    
    async def _init_external_learning_async(self) -> Optional[ExternalLearning]:
        """Initialize external learning component."""
        try:
            external_learning = ExternalLearning(base_dir='.')
            self.logger.info("ExternalLearning initialized")
            return external_learning
        except Exception as e:
            self.logger.warning(f"Could not initialize ExternalLearning: {e}")
            return None
    
    async def _init_github_integration_async(self) -> Optional[GitHubIntegration]:
        """Initialize GitHub integration component."""
        try:
            github_integration = GitHubIntegration()
            self.logger.info("GitHubIntegration initialized")
            return github_integration
        except Exception as e:
            self.logger.warning(f"Could not initialize GitHubIntegration: {e}")
            return None


class ConfigLoader:
    """Load and manage configuration."""
    
    DEFAULT_CONFIG_PATH = Path('config/deep_before_wide.json')
    
    DEFAULT_VALUES = {
        'max_repos_per_question': 3,
        'max_files_per_repo': 10,
        'cache_duration_hours': 24
    }
    
    @staticmethod
    def load(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration from file or use provided config."""
        if config is not None:
            return config
        
        if ConfigLoader.DEFAULT_CONFIG_PATH.exists():
            with open(ConfigLoader.DEFAULT_CONFIG_PATH, 'r') as f:
                full_config = json.load(f)
                return full_config.get('github_learning', {})
        
        return {}
    
    @staticmethod
    def get_value(config: Dict[str, Any], key: str) -> Any:
        """Get configuration value with default fallback."""
        return config.get(key, ConfigLoader.DEFAULT_VALUES.get(key))


class AutonomousLearnerV2(AutonomousLearnerV1):
    """Enhanced autonomous learning with GitHub integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.config = ConfigLoader.load(config)
        self._init_config_values()
        self.components = {}

    def _init_config_values(self):
        """Initialize configuration values."""
        self.max_repos_per_question = ConfigLoader.get_value(self.config, 'max_repos_per_question')
        self.max_files_per_repo