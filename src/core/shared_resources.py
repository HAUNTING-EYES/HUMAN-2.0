#!/usr/bin/env python3
"""
HUMAN 2.0 - Shared Resources Manager
Unified access to all shared data structures and services for multi-agent system.

Provides:
- ChromaDB (semantic search)
- Dependency Graph (code relationships)
- Knowledge Graph (semantic understanding)
- Pattern Library (reusable patterns)
- Model Cache (LLM response caching)
- Metrics Database (centralized metrics)
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import networkx as nx

# Import existing V2 components
try:
    from .dependency_analyzer import DependencyAnalyzer
    from .code_embedder import CodeEmbedder
    from .meta_learner import MetaLearner
    from .semantic_code_model import SemanticCodeModel  # PHASE 2
    from .thought_trace import ThoughtTraceManager  # PHASE 4
except ImportError:
    # Running as __main__, use absolute imports
    from src.core.dependency_analyzer import DependencyAnalyzer
    from src.core.code_embedder import CodeEmbedder
    from src.core.meta_learner import MetaLearner
    from src.core.semantic_code_model import SemanticCodeModel  # PHASE 2
    from src.core.thought_trace import ThoughtTraceManager  # PHASE 4


@dataclass
class Pattern:
    """Reusable improvement pattern"""
    pattern_id: str
    name: str
    description: str
    category: str  # e.g., "refactoring", "testing", "performance"
    code_template: str
    success_count: int = 0
    failure_count: int = 0
    avg_improvement: float = 0.0
    tags: List[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def success_rate(self) -> float:
        """Calculate pattern success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    node_id: str
    node_type: str  # "concept", "function", "class", "module", "pattern"
    content: str
    metadata: Dict[str, Any]
    connections: Set[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.connections is None:
            self.connections = set()
        if self.created_at is None:
            self.created_at = datetime.now()


class SharedResources:
    """
    Unified resource manager for multi-agent system.

    All agents access shared resources through this manager for:
    - Consistency: Single source of truth
    - Efficiency: Shared caches and indexes
    - Safety: Centralized validation
    """

    def __init__(self,
                 root_dir: str = '.',
                 chroma_dir: str = "data/unified_chroma_db",
                 knowledge_graph_path: str = "data/knowledge_graph.json",
                 pattern_library_path: str = "data/pattern_library.json",
                 model_cache_path: str = "data/model_cache.json",
                 metrics_db_path: str = "data/metrics_db.json",
                 semantic_model_path: str = "data/semantic_model.json"):
        """
        Initialize shared resources manager.

        Args:
            root_dir: Project root directory
            chroma_dir: ChromaDB persistence directory
            knowledge_graph_path: Knowledge graph storage
            pattern_library_path: Pattern library storage
            model_cache_path: Model cache storage
            metrics_db_path: Metrics database storage
        """
        self.logger = logging.getLogger(__name__)
        self.root_dir = Path(root_dir).resolve()

        # Data directories
        self.data_dir = self.root_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize existing V2 components
        self.logger.info("Initializing V2 components...")
        self.dependency_analyzer = DependencyAnalyzer(root_dir=str(self.root_dir))
        self.code_embedder = CodeEmbedder(chroma_dir=chroma_dir)
        self.meta_learner = MetaLearner()

        # Initialize new multi-agent components
        self.logger.info("Initializing multi-agent components...")

        # Knowledge Graph (NEW)
        self.knowledge_graph_path = Path(knowledge_graph_path)
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self._load_knowledge_graph()

        # Pattern Library (NEW)
        self.pattern_library_path = Path(pattern_library_path)
        self.pattern_library: Dict[str, Pattern] = {}
        self._load_pattern_library()

        # Model Cache (NEW) - Cache LLM responses to save API calls
        self.model_cache_path = Path(model_cache_path)
        self.model_cache: Dict[str, Any] = {}
        self._load_model_cache()

        # Metrics Database (NEW) - Centralized metrics tracking
        self.metrics_db_path = Path(metrics_db_path)
        self.metrics_db: Dict[str, Any] = {}
        self._load_metrics_db()

        # Semantic Code Model (PHASE 2) - Deep code understanding
        self.semantic_model_path = Path(semantic_model_path)
        self.semantic_model = SemanticCodeModel()
        self._load_semantic_model()

        # Thought Trace Manager (PHASE 4) - Agent reasoning transparency
        self.thought_trace_manager = ThoughtTraceManager(trace_dir="data/thought_traces")

        self.logger.info(f"SharedResources initialized with root: {self.root_dir}")
        self.logger.info(f"Knowledge Graph: {len(self.knowledge_graph)} nodes")
        self.logger.info(f"Pattern Library: {len(self.pattern_library)} patterns")
        self.logger.info(f"Model Cache: {len(self.model_cache)} entries")
        self.logger.info(f"Semantic Model: {len(self.semantic_model.component_understanding)} components")
        self.logger.info(f"Thought Trace Manager: Initialized")

    # ========== V2 Component Access ==========

    def get_dependencies(self, file_path: str) -> List[str]:
        """Get dependencies of a file (what it imports)"""
        return self.dependency_analyzer.get_dependencies(file_path)

    def get_reverse_dependencies(self, file_path: str) -> List[str]:
        """Get reverse dependencies (what imports this file)"""
        return self.dependency_analyzer.get_reverse_dependencies(file_path)

    def get_file_criticality(self, file_path: str) -> float:
        """Get file criticality score (0-1)"""
        return self.dependency_analyzer.calculate_criticality(file_path)

    def get_dependency_graph(self) -> nx.DiGraph:
        """Get the complete dependency graph"""
        return self.dependency_analyzer.graph

    def search_similar_code(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar code using ChromaDB semantic search"""
        return self.code_embedder.find_similar_code(query, n_results=n_results)

    def embed_file(self, file_path: str):
        """Embed a single file in ChromaDB"""
        self.code_embedder.embed_file(file_path)

    def record_improvement_outcome(self, outcome: Dict[str, Any]):
        """Record improvement outcome for meta-learning"""
        self.meta_learner.record_outcome(outcome)

    # ========== Knowledge Graph ==========

    def add_knowledge_node(self, node: KnowledgeNode):
        """Add node to knowledge graph"""
        self.knowledge_graph[node.node_id] = node
        self.logger.debug(f"Added knowledge node: {node.node_id}")

    def get_knowledge_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get knowledge node by ID"""
        return self.knowledge_graph.get(node_id)

    def connect_knowledge_nodes(self, node_id_1: str, node_id_2: str):
        """Create bidirectional connection between knowledge nodes"""
        if node_id_1 in self.knowledge_graph:
            self.knowledge_graph[node_id_1].connections.add(node_id_2)
        if node_id_2 in self.knowledge_graph:
            self.knowledge_graph[node_id_2].connections.add(node_id_1)
        self.logger.debug(f"Connected nodes: {node_id_1} <-> {node_id_2}")

    def search_knowledge_graph(self, query: str, node_type: Optional[str] = None) -> List[KnowledgeNode]:
        """Search knowledge graph by content or type"""
        results = []
        for node in self.knowledge_graph.values():
            if node_type and node.node_type != node_type:
                continue
            if query.lower() in node.content.lower():
                results.append(node)
        return results

    def _load_knowledge_graph(self):
        """Load knowledge graph from disk"""
        if self.knowledge_graph_path.exists():
            try:
                with open(self.knowledge_graph_path, 'r') as f:
                    data = json.load(f)
                    for node_id, node_data in data.items():
                        # Convert connections from list to set
                        if 'connections' in node_data:
                            node_data['connections'] = set(node_data['connections'])
                        # Convert timestamp string to datetime
                        if 'created_at' in node_data:
                            node_data['created_at'] = datetime.fromisoformat(node_data['created_at'])
                        self.knowledge_graph[node_id] = KnowledgeNode(**node_data)
                self.logger.info(f"Loaded {len(self.knowledge_graph)} knowledge nodes")
            except Exception as e:
                self.logger.error(f"Error loading knowledge graph: {e}")
                self.knowledge_graph = {}

    def save_knowledge_graph(self):
        """Save knowledge graph to disk"""
        try:
            data = {}
            for node_id, node in self.knowledge_graph.items():
                node_dict = asdict(node)
                # Convert set to list for JSON serialization
                node_dict['connections'] = list(node.connections)
                # Convert datetime to ISO string
                node_dict['created_at'] = node.created_at.isoformat()
                data[node_id] = node_dict

            with open(self.knowledge_graph_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.knowledge_graph)} knowledge nodes")
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {e}")

    # ========== Pattern Library ==========

    def add_pattern(self, pattern: Pattern):
        """Add pattern to library"""
        self.pattern_library[pattern.pattern_id] = pattern
        self.logger.debug(f"Added pattern: {pattern.name}")

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID"""
        return self.pattern_library.get(pattern_id)

    def search_patterns(self, category: Optional[str] = None,
                       min_success_rate: float = 0.0,
                       tags: Optional[List[str]] = None) -> List[Pattern]:
        """Search patterns by criteria"""
        results = []
        for pattern in self.pattern_library.values():
            # Filter by category
            if category and pattern.category != category:
                continue
            # Filter by success rate
            if pattern.success_rate < min_success_rate:
                continue
            # Filter by tags
            if tags and not any(tag in pattern.tags for tag in tags):
                continue
            results.append(pattern)

        # Sort by success rate
        results.sort(key=lambda p: p.success_rate, reverse=True)
        return results

    def update_pattern_outcome(self, pattern_id: str, success: bool, improvement: float = 0.0):
        """Update pattern outcome statistics"""
        if pattern_id in self.pattern_library:
            pattern = self.pattern_library[pattern_id]
            if success:
                pattern.success_count += 1
            else:
                pattern.failure_count += 1

            # Update rolling average improvement
            total = pattern.success_count + pattern.failure_count
            pattern.avg_improvement = ((pattern.avg_improvement * (total - 1)) + improvement) / total

            self.logger.debug(f"Updated pattern {pattern.name}: success_rate={pattern.success_rate:.2f}")

    def _load_pattern_library(self):
        """Load pattern library from disk"""
        if self.pattern_library_path.exists():
            try:
                with open(self.pattern_library_path, 'r') as f:
                    data = json.load(f)
                    for pattern_id, pattern_data in data.items():
                        # Convert timestamp string to datetime
                        if 'created_at' in pattern_data:
                            pattern_data['created_at'] = datetime.fromisoformat(pattern_data['created_at'])
                        self.pattern_library[pattern_id] = Pattern(**pattern_data)
                self.logger.info(f"Loaded {len(self.pattern_library)} patterns")
            except Exception as e:
                self.logger.error(f"Error loading pattern library: {e}")
                self.pattern_library = {}

    def save_pattern_library(self):
        """Save pattern library to disk"""
        try:
            data = {}
            for pattern_id, pattern in self.pattern_library.items():
                pattern_dict = asdict(pattern)
                # Convert datetime to ISO string
                pattern_dict['created_at'] = pattern.created_at.isoformat()
                data[pattern_id] = pattern_dict

            with open(self.pattern_library_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.pattern_library)} patterns")
        except Exception as e:
            self.logger.error(f"Error saving pattern library: {e}")

    # ========== Model Cache ==========

    def get_cached_response(self, prompt: str, model: str = "claude-3.5-sonnet") -> Optional[str]:
        """Get cached LLM response if available"""
        cache_key = self._make_cache_key(prompt, model)
        entry = self.model_cache.get(cache_key)
        if entry:
            self.logger.debug(f"Cache hit for prompt hash: {cache_key[:8]}...")
            return entry['response']
        return None

    def cache_response(self, prompt: str, response: str, model: str = "claude-3.5-sonnet"):
        """Cache LLM response"""
        cache_key = self._make_cache_key(prompt, model)
        self.model_cache[cache_key] = {
            'prompt': prompt,
            'response': response,
            'model': model,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.debug(f"Cached response for prompt hash: {cache_key[:8]}...")

    def _make_cache_key(self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model"""
        content = f"{model}::{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_model_cache(self):
        """Load model cache from disk"""
        if self.model_cache_path.exists():
            try:
                with open(self.model_cache_path, 'r') as f:
                    self.model_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.model_cache)} cached responses")
            except Exception as e:
                self.logger.error(f"Error loading model cache: {e}")
                self.model_cache = {}

    def save_model_cache(self):
        """Save model cache to disk"""
        try:
            with open(self.model_cache_path, 'w') as f:
                json.dump(self.model_cache, f, indent=2)
            self.logger.info(f"Saved {len(self.model_cache)} cached responses")
        except Exception as e:
            self.logger.error(f"Error saving model cache: {e}")

    # ========== Metrics Database ==========

    def record_metric(self, category: str, name: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric"""
        if category not in self.metrics_db:
            self.metrics_db[category] = {}

        if name not in self.metrics_db[category]:
            self.metrics_db[category][name] = []

        self.metrics_db[category][name].append({
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })

    def get_metrics(self, category: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics by category and optionally by name"""
        if category not in self.metrics_db:
            return {}

        if name:
            return self.metrics_db[category].get(name, [])

        return self.metrics_db[category]

    def get_latest_metric(self, category: str, name: str) -> Optional[Any]:
        """Get the latest value for a metric"""
        metrics = self.get_metrics(category, name)
        if metrics:
            return metrics[-1]['value']
        return None

    def _load_metrics_db(self):
        """Load metrics database from disk"""
        if self.metrics_db_path.exists():
            try:
                with open(self.metrics_db_path, 'r') as f:
                    self.metrics_db = json.load(f)
                total_metrics = sum(len(metrics) for category in self.metrics_db.values() for metrics in category.values())
                self.logger.info(f"Loaded metrics database: {total_metrics} total metrics")
            except Exception as e:
                self.logger.error(f"Error loading metrics database: {e}")
                self.metrics_db = {}

    def save_metrics_db(self):
        """Save metrics database to disk"""
        try:
            with open(self.metrics_db_path, 'w') as f:
                json.dump(self.metrics_db, f, indent=2)
            total_metrics = sum(len(metrics) for category in self.metrics_db.values() for metrics in category.values())
            self.logger.info(f"Saved metrics database: {total_metrics} total metrics")
        except Exception as e:
            self.logger.error(f"Error saving metrics database: {e}")

    # ========== Semantic Model (PHASE 2) ==========

    def _load_semantic_model(self):
        """Load semantic model from disk"""
        try:
            if self.semantic_model_path.exists():
                self.semantic_model.load(str(self.semantic_model_path))
                self.logger.info(f"Loaded semantic model: {len(self.semantic_model.component_understanding)} components")
            else:
                self.logger.info("No existing semantic model found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading semantic model: {e}")

    def save_semantic_model(self):
        """Save semantic model to disk"""
        try:
            self.semantic_model.save(str(self.semantic_model_path))
            self.logger.info(f"Saved semantic model: {len(self.semantic_model.component_understanding)} components")
        except Exception as e:
            self.logger.error(f"Error saving semantic model: {e}")

    def get_component_understanding(self, file_path: str):
        """Get semantic understanding of a component"""
        return self.semantic_model.component_understanding.get(file_path)

    def analyze_component_semantics(self, file_path: str, code: str):
        """Analyze component semantics"""
        return self.semantic_model.analyze_component(file_path, code)

    # ========== Persistence ==========

    def save_all(self):
        """Save all resources to disk"""
        self.logger.info("Saving all shared resources...")
        self.save_knowledge_graph()
        self.save_pattern_library()
        self.save_model_cache()
        self.save_metrics_db()
        self.save_semantic_model()  # PHASE 2
        self.logger.info("All resources saved")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about shared resources"""
        return {
            'dependency_graph': {
                'nodes': self.dependency_analyzer.graph.number_of_nodes(),
                'edges': self.dependency_analyzer.graph.number_of_edges()
            },
            'chromadb': {
                'codebase_count': self.code_embedder.codebase_collection.count(),
                'improvements_count': self.code_embedder.improvements_collection.count(),
                'external_knowledge_count': self.code_embedder.external_knowledge_collection.count()
            },
            'knowledge_graph': {
                'nodes': len(self.knowledge_graph)
            },
            'pattern_library': {
                'total_patterns': len(self.pattern_library),
                'avg_success_rate': sum(p.success_rate for p in self.pattern_library.values()) / len(self.pattern_library) if self.pattern_library else 0.0
            },
            'model_cache': {
                'cached_responses': len(self.model_cache)
            },
            'metrics_db': {
                'categories': len(self.metrics_db),
                'total_metrics': sum(len(metrics) for category in self.metrics_db.values() for metrics in category.values())
            }
        }


# Singleton instance
_shared_resources = None

def get_shared_resources() -> SharedResources:
    """Get the singleton shared resources instance"""
    global _shared_resources
    if _shared_resources is None:
        _shared_resources = SharedResources()
    return _shared_resources


if __name__ == "__main__":
    # Test shared resources
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    resources = SharedResources()

    # Test knowledge graph
    node1 = KnowledgeNode(
        node_id="node1",
        node_type="concept",
        content="Python async/await pattern",
        metadata={"category": "concurrency"}
    )
    resources.add_knowledge_node(node1)

    node2 = KnowledgeNode(
        node_id="node2",
        node_type="pattern",
        content="Event-driven architecture",
        metadata={"category": "architecture"}
    )
    resources.add_knowledge_node(node2)

    resources.connect_knowledge_nodes("node1", "node2")

    # Test pattern library
    pattern = Pattern(
        pattern_id="pattern1",
        name="Async Refactoring",
        description="Convert sync code to async",
        category="refactoring",
        code_template="async def {function_name}():\n    ...",
        tags=["async", "performance"]
    )
    resources.add_pattern(pattern)
    resources.update_pattern_outcome("pattern1", success=True, improvement=0.25)

    # Test model cache
    resources.cache_response("test prompt", "test response")
    cached = resources.get_cached_response("test prompt")
    print(f"Cached response: {cached}")

    # Test metrics
    resources.record_metric("test_coverage", "overall", 0.75)
    resources.record_metric("complexity", "avg", 8.5)

    # Save all
    resources.save_all()

    # Get stats
    stats = resources.get_stats()
    print(f"\nShared Resources Stats:")
    print(json.dumps(stats, indent=2))
