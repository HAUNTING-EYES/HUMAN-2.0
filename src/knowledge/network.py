"""
HUMAN 2.0 - Knowledge Network
Graph-based knowledge representation for exponential growth tracking.

Features:
- Knowledge nodes with topics and content
- Edges representing relationships
- Gap identification for learning suggestions
- Growth rate calculation
- Persistence to JSON
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class RelationType(Enum):
    """Types of relationships between knowledge nodes."""
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    EXTENDS = "extends"
    CONTRADICTS = "contradicts"
    EXAMPLE_OF = "example_of"
    PART_OF = "part_of"


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""
    node_id: str
    topic: str
    content: str
    source: str
    confidence: float
    importance: float
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_count: int = 0
    last_accessed: datetime = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'topic': self.topic,
            'content': self.content,
            'source': self.source,
            'confidence': self.confidence,
            'importance': self.importance,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'accessed_count': self.accessed_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_accessed'):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


@dataclass
class KnowledgeEdge:
    """An edge (relationship) between knowledge nodes."""
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type.value,
            'strength': self.strength,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
        """Create from dictionary."""
        data['relation_type'] = RelationType(data['relation_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class KnowledgeData:
    """Parameter object for adding knowledge."""
    topic: str
    content: str
    source: str = "internal"
    confidence: float = 0.8
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)


class KnowledgeNetwork:
    """
    Graph-based knowledge representation.

    Tracks:
    - What has been learned (nodes)
    - How knowledge is connected (edges)
    - What gaps exist (missing connections)
    - Growth rate over time

    Enables:
    - Smart learning suggestions
    - Knowledge retrieval
    - Exponential growth tracking
    """

    def __init__(self, storage_path: str = "data/knowledge_network.json"):
        """
        Initialize knowledge network.

        Args:
            storage_path: Path for persistence
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []

        self._topic_index: Dict[str, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)

        self._growth_history: List[Tuple[datetime, int]] = []

        self._load()

        self.logger.info(f"KnowledgeNetwork initialized with {len(self.nodes)} nodes")

    def add_knowledge(
        self,
        topic: str,
        content: str,
        source: str = "internal",
        confidence: float = 0.8,
        importance: float = 0.5,
        tags: List[str] = None,
        related_topics: List[str] = None
    ) -> str:
        """
        Add new knowledge to the network.

        Args:
            topic: Topic/title of the knowledge
            content: Knowledge content
            source: Where this knowledge came from
            confidence: How confident we are (0-1)
            importance: How important this is (0-1)
            tags: Tags for categorization
            related_topics: Related topic strings to link to

        Returns:
            Node ID of created node
        """
        data = KnowledgeData(
            topic=topic,
            content=content,
            source=source,
            confidence=confidence,
            importance=importance,
            tags=tags or [],
            related_topics=related_topics or []
        )
        return self._add_knowledge_from_data(data)

    def _add_knowledge_from_data(self, data: KnowledgeData) -> str:
        """Internal method to add knowledge from data object."""
        node_id = self._generate_node_id(data.topic)
        node = self._create_node(node_id, data)
        
        self._add_node_to_graph(node, data.related_topics)
        self._track_growth()
        self._save()

        self.logger.debug(f"Added knowledge node: {node_id}")
        return node_id

    def _generate_node_id(self, topic: str) -> str:
        """Generate unique node ID."""
        return f"{topic.lower().replace(' ', '_')}_{len(self.nodes)}"

    def _create_node(self, node_id: str, data: KnowledgeData) -> KnowledgeNode:
        """Create a knowledge node."""
        return KnowledgeNode(
            node_id=node_id,
            topic=data.topic,
            content=data.content,
            source=data.source,
            confidence=data.confidence,
            importance=data.importance,
            tags=data.tags
        )

    def _add_node_to_graph(self, node: KnowledgeNode, related_topics: List[str]):
        """Add node to graph and update indexes."""
        self.nodes[node.node_id] = node
        self._update_indexes(node)
        self._create_relations(node.node_id, related_topics)
        self._discover_connections(node)

    def _update_indexes(self, node: KnowledgeNode):
        """Update topic and tag indexes."""
        self._topic_index[node.topic.lower()].add(node.node_id)
        for tag in node.tags:
            self._tag_index[tag.lower()].add(node.node_id)

    def _create_relations(self, source_id: str, related_topics: List[str]):
        """Create relations to related topics."""
        for topic in related_topics:
            self._create_relation(source_id, topic)

    def _track_growth(self):
        """Track growth history."""
        self._growth_history.append((datetime.now(), len(self.nodes)))

    def _create_relation(self, source_id: str, target_topic: str):
        """Create a relation between nodes."""
        target_ids = self._topic_index.get(target_topic.lower(), set())

        for target_id in target_ids:
            if target_id != source_id:
                self._add_edge(source_id, target_id, RelationType.RELATED_TO, 0.5)

    def _add_edge(self, source_id: str, target_id: str, relation_type: RelationType, strength: float):
        """Add an edge to the graph."""
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength
        )
        self.edges.append(edge)
        self._adjacency[source_id].add(target_id)
        self._adjacency[target_id].add(source_id)

    def _discover_connections(self, node: KnowledgeNode):
        """Auto-discover connections based on content similarity."""
        keywords = self._extract_keywords(node)

        for other_id, other_node in self.nodes.items():
            if other_id == node.node_id:
                continue

            if self._should_connect(node, other_node, keywords):
                strength = self._calculate_connection_strength(keywords, other_node)
                self._add_edge(node.node_id, other_id, RelationType.RELATED_TO, strength)

    def _extract_keywords(self, node: KnowledgeNode) -> Set[str]:
        """Extract keywords from node."""
        return set(node.topic.lower().split() + node.tags)

    def _should_connect(self, node: KnowledgeNode, other_node: KnowledgeNode, keywords: Set[str]) -> bool:
        """Check if nodes should be connected."""
        other_keywords = self._extract_keywords(other_node)
        overlap = keywords & other_keywords
        return len(overlap) >= 2

    def _calculate_connection_strength(self, keywords: Set[str], other_node: KnowledgeNode) -> float:
        """Calculate connection strength between nodes."""
        other_keywords = self._extract_keywords(other_node)
        overlap = keywords & other_keywords
        return min(1.0, len(overlap) / 5.0)

    def get_knowledge(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a knowledge node by ID."""
        node = self.nodes.get(node_id)
        if node:
            node.accessed_count += 1
            node.last_accessed = datetime.now()
        return node

    def search_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeNode]:
        """
        Search for knowledge nodes.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching nodes
        """
        query_terms = query.lower().split()
        results = [
            (self._calculate_search_score(node, query_terms), node)
            for node in self.nodes.values()
        ]
        
        scored_results = [(score, node) for score, node in results if score > 0]
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored_results[:limit]]

    def _calculate_search_score(self, node: KnowledgeNode, query_terms: List[str]) -> float:
        """Calculate search score for a node."""
        score = 0
        score += self._score_topic_match(node, query_terms)
        score += self._score_tag_match(node, query_terms)
        score += self._score_content_match(node, query_terms)
        return score

    def _score_topic_match(self, node: KnowledgeNode, query_terms: List[str]) -> float:
        """Score topic matches."""
        return sum(2 for term in query_terms if term in node.topic.lower())

    def _score_tag_match(self, node: KnowledgeNode, query_terms: List[str]) -> float:
        """Score tag matches."""
        node_tags_lower = [t.lower() for t in node.tags]
        return sum(1 for term in query_terms if term in node_tags_lower)

    def _score_content_match(self, node: KnowledgeNode, query_terms: List[str]) -> float:
        """Score content matches."""
        node_content_lower = node.content.lower()
        return sum(0.5 for term in query_terms if term in node_content_lower)

    def get_related(self, node_id: str, depth: int = 2) -> List[KnowledgeNode]:
        """
        Get related knowledge nodes (BFS traversal).

        Args:
            node_id: Starting node
            depth: How many hops to traverse

        Returns:
            List of related nodes
        """
        if node_id not in self.nodes:
            return []

        visited = {node_id}
        current_level = {node_id}
        result = []

        for _ in range(depth):
            next_level = self._get_next_level(current_level, visited, result)
            current_level = next_level

        return result

    def _get_next_level(self, current_level: Set[str], visited: Set[str], result: List[KnowledgeNode]) -> Set[str]:
        """Get next level of nodes in BFS traversal."""
        next_level = set()
        for current_id in current_level:
            neighbors = self._adjacency.get(current_id, set())
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_level.add(neighbor_id)
                    if neighbor_id in self.nodes:
                        result.append(self.nodes[neighbor_id])
        return next_level

    def identify_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify gaps in knowledge (topics that should be connected but aren't).

        Returns:
            List of gap descriptions with suggested topics
        """
        gaps = []
        gaps.extend(self._find_isolated_nodes())
        gaps.extend(self._find_tag_gaps())
        return sorted(gaps, key=lambda x: x['priority'], reverse=True)[:10]

    def _find_isolated_nodes(self) -> List[Dict[str, Any]]:
        """Find isolated important nodes."""
        gaps = []
        for node_id, node in self.nodes.items():
            connections = len(self._adjacency.get(node_id, set()))
            if connections < 2 and node.importance > 0.5:
                gaps.append({
                    'type': 'isolated_important',
                    'node': node.topic,
                    'suggestion': f"Learn more about topics related to {node.topic}",
                    'priority': node.importance
                })
        return gaps

    def _find_tag_gaps(self) -> List[Dict[str, Any]]:
        """Find missing connections between tags."""
        tag_pairs = self._count_tag_pairs()
        gaps = []

        for (tag1, tag2), count in tag_pairs.items():
            if self._is_tag_gap(tag1, tag2, count):
                gaps.append({
                    'type': 'tag_gap',
                    'tags': [tag1, tag2],
                    'suggestion': f"Learn about intersection of {tag1} and {tag2}",
                    'priority': 0.6
                })