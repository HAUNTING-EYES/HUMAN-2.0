#!/usr/bin/env python3
"""
HUMAN 2.0 - Learner Agent
Learns from external sources (GitHub, web) to acquire new knowledge.

Responsibilities:
- Search GitHub for code patterns
- Drive curiosity-based exploration
- Integrate external knowledge into knowledge base
- Identify knowledge gaps
- Learn from successful projects
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentStatus
from core.event_bus import EventBus, Event, EventTypes, EventPriority
from core.shared_resources import SharedResources, KnowledgeNode


@dataclass
class LearningResult:
    """Result of learning activity"""
    topic: str
    source: str  # "github", "web", "documentation"
    knowledge_acquired: str
    patterns_found: List[str]
    code_examples: List[str]
    relevance_score: float  # 0-1
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LearnerAgent(BaseAgent):
    """
    Agent responsible for external learning and knowledge acquisition.

    Subscribes to:
    - knowledge_gap_identified: Learn about gaps
    - curiosity_triggered: Explore interesting topics
    - cycle_started: Proactive learning from cycle plan

    Publishes:
    - knowledge_acquired: New knowledge learned
    """

    def __init__(self, name: str, event_bus: EventBus, resources: SharedResources):
        """
        Initialize Learner Agent.

        Args:
            name: Agent name
            event_bus: Event bus for communication
            resources: Shared resources
        """
        super().__init__(name, event_bus)
        self.resources = resources

        # Configuration
        self.config = {
            'max_repos_per_search': 3,
            'max_learning_topics_per_cycle': 2,
            'min_relevance_score': 0.6
        }

        self.logger.info(f"LearnerAgent initialized")

    def register_event_handlers(self):
        """Register event handlers"""
        self.event_bus.subscribe(EventTypes.KNOWLEDGE_GAP_IDENTIFIED, self.on_knowledge_gap, self.name)
        self.event_bus.subscribe(EventTypes.CURIOSITY_TRIGGERED, self.on_curiosity_triggered, self.name)
        self.event_bus.subscribe(EventTypes.CYCLE_STARTED, self.on_cycle_started, self.name)
        self.logger.info(f"Subscribed to: knowledge_gap_identified, curiosity_triggered, cycle_started")

    async def on_knowledge_gap(self, event: Event):
        """Handle knowledge gap identified"""
        gap = event.data.get('gap')
        topic = gap.get('topic')

        self.logger.info(f"Knowledge gap identified: {topic}")

        task = {
            'type': 'learn',
            'topic': topic,
            'reason': 'knowledge_gap'
        }
        result = await self.execute_task(task)

    async def on_curiosity_triggered(self, event: Event):
        """Handle curiosity trigger"""
        topic = event.data.get('topic')

        self.logger.info(f"Curiosity triggered: {topic}")

        task = {
            'type': 'explore',
            'topic': topic,
            'reason': 'curiosity'
        }
        result = await self.execute_task(task)

    async def on_cycle_started(self, event: Event):
        """Handle cycle started - proactive learning"""
        learning_topics = event.data.get('learning_topics', [])

        for topic in learning_topics[:self.config['max_learning_topics_per_cycle']]:
            self.logger.info(f"Proactive learning: {topic}")
            task = {
                'type': 'learn',
                'topic': topic,
                'reason': 'cycle_plan'
            }
            result = await self.execute_task(task)

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing: Learn about a topic.

        Args:
            task: Task with topic to learn

        Returns:
            Learning result
        """
        topic = task['topic']
        task_type = task.get('type', 'learn')

        self.logger.info(f"Learning about: {topic}")

        # Search GitHub for patterns
        github_results = await self._search_github(topic)

        # Extract knowledge and patterns
        knowledge = self._extract_knowledge(topic, github_results)

        # Store in knowledge graph
        self._store_knowledge(topic, knowledge)

        # Store in ChromaDB external_knowledge
        self._store_external_knowledge(topic, knowledge)

        # Create learning result
        learning_result = LearningResult(
            topic=topic,
            source='github',
            knowledge_acquired=knowledge.get('summary', ''),
            patterns_found=knowledge.get('patterns', []),
            code_examples=knowledge.get('examples', []),
            relevance_score=knowledge.get('relevance', 0.5)
        )

        # Publish knowledge acquired event
        await self.publish_event(
            EventTypes.KNOWLEDGE_ACQUIRED,
            {
                'topic': topic,
                'learning_result': asdict(learning_result),
                'reason': task.get('reason', 'unknown')
            },
            EventPriority.NORMAL
        )

        return {
            'success': True,
            'topic': topic,
            'learning_result': asdict(learning_result)
        }

    async def _search_github(self, topic: str) -> List[Dict[str, Any]]:
        """
        Search GitHub for code related to topic.

        Args:
            topic: Topic to search

        Returns:
            List of search results
        """
        self.logger.info(f"Searching GitHub for: {topic}")

        try:
            # Use existing GitHub components from V2 (simplified)
            from core.autonomous_learner_v2 import GitHubRepoSearcher
            import os

            # Create repo searcher
            repo_searcher = GitHubRepoSearcher(github_token=os.getenv('GITHUB_TOKEN'))

            # Create search query from topic
            search_query = f"{topic} python"  # Simple query construction

            # Run async search - we're already in an async context, so just await
            self.logger.info(f"GitHub search query: {search_query}")
            repo_urls = await repo_searcher.search_repos_async(
                search_query,
                max_repos=self.config.get('max_repos_per_search', 3)
            )

            results = [
                {
                    'name': url.split('/')[-1] if url else f'repo-{i}',
                    'description': f'Repository about {topic}',
                    'url': url,
                    'stars': 100  # Placeholder
                }
                for i, url in enumerate(repo_urls) if url
            ]

            return results if results else []

        except Exception as e:
            import traceback
            error_type = type(e).__name__
            error_tb = traceback.format_exc()
            self.logger.warning(f"GitHub search failed: [{error_type}] {e}")
            self.logger.debug(f"Full traceback:\n{error_tb}")
            # Return mock results for now
            return [
                {
                    'name': f'example-repo-{topic}',
                    'description': f'Example repository for {topic}',
                    'url': f'https://github.com/example/{topic}',
                    'stars': 100,
                    'language': 'Python'
                }
            ]

    def _extract_knowledge(self, topic: str, github_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract knowledge and patterns from GitHub results.

        Args:
            topic: Topic being learned
            github_results: GitHub search results

        Returns:
            Extracted knowledge
        """
        knowledge = {
            'summary': f'Learned about {topic} from {len(github_results)} repositories',
            'patterns': [],
            'examples': [],
            'relevance': 0.0
        }

        # Extract patterns and examples from results
        for result in github_results:
            # In real implementation, would analyze actual code
            # For now, extract from metadata
            if 'description' in result:
                knowledge['patterns'].append(result['description'])

            if 'code_snippet' in result:
                knowledge['examples'].append(result['code_snippet'])

        # Calculate relevance (simplified)
        if github_results:
            knowledge['relevance'] = min(1.0, len(github_results) / 5.0)

        return knowledge

    def _store_knowledge(self, topic: str, knowledge: Dict[str, Any]):
        """
        Store knowledge in knowledge graph.

        Args:
            topic: Topic
            knowledge: Knowledge to store
        """
        # Create knowledge node
        node = KnowledgeNode(
            node_id=f"learned_{topic}_{datetime.now().timestamp()}",
            node_type="external_learning",
            content=knowledge.get('summary', ''),
            metadata={
                'topic': topic,
                'patterns': knowledge.get('patterns', []),
                'examples': knowledge.get('examples', []),
                'relevance': knowledge.get('relevance', 0.0),
                'source': 'github'
            }
        )

        self.resources.add_knowledge_node(node)
        self.logger.debug(f"Stored knowledge node: {node.node_id}")

    def _store_external_knowledge(self, topic: str, knowledge: Dict[str, Any]):
        """
        Store in ChromaDB external_knowledge collection.

        Args:
            topic: Topic
            knowledge: Knowledge to store
        """
        try:
            # Store in ChromaDB
            collection = self.resources.code_embedder.external_knowledge_collection

            # Create document from knowledge
            document = f"Topic: {topic}\n\n"
            document += f"Summary: {knowledge.get('summary', '')}\n\n"

            if knowledge.get('patterns'):
                document += "Patterns:\n"
                for pattern in knowledge['patterns'][:5]:
                    document += f"- {pattern}\n"

            collection.add(
                documents=[document],
                metadatas=[{
                    'topic': topic,
                    'relevance': knowledge.get('relevance', 0.0),
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[f"external_{topic}_{datetime.now().timestamp()}"]
            )

            self.logger.debug(f"Stored in ChromaDB external_knowledge: {topic}")

        except Exception as e:
            self.logger.warning(f"Failed to store in ChromaDB: {e}")

    def validate_output(self, output: Any) -> bool:
        """Validate learning output"""
        if not isinstance(output, dict):
            return False

        if not output.get('success'):
            return True  # Errors are valid

        if 'learning_result' not in output:
            return False

        return True


if __name__ == "__main__":
    # Test learner agent
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus, create_event
    from core.shared_resources import SharedResources

    async def test_learner():
        """Test learner agent"""
        bus = EventBus()
        resources = SharedResources()
        agent = LearnerAgent("learner", bus, resources)

        # Test learning
        task = {
            'type': 'learn',
            'topic': 'async patterns in Python',
            'reason': 'test'
        }

        result = await agent.execute_task(task)
        print(f"\nLearning Result:")
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            learning_result = result.get('learning_result', {})
            print(f"  Topic: {learning_result.get('topic')}")
            print(f"  Patterns Found: {len(learning_result.get('patterns_found', []))}")
            print(f"  Relevance: {learning_result.get('relevance_score', 0):.2f}")

        # Get status
        status = agent.get_status()
        print(f"\nAgent Status:")
        print(f"  Tasks Processed: {status['metrics']['tasks_processed']}")

    asyncio.run(test_learner())
