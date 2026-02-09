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

# Consciousness systems
try:
    from consciousness.curiosity import CuriosityEngine
    HAS_CURIOSITY_ENGINE = True
except ImportError:
    try:
        from src.consciousness.curiosity import CuriosityEngine
        HAS_CURIOSITY_ENGINE = True
    except ImportError:
        HAS_CURIOSITY_ENGINE = False
        CuriosityEngine = None


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


@dataclass
class ResourcePaths:
    """Configuration for resource storage paths"""
    root_dir: Path
    chroma_dir: str
    knowledge_graph_path: Path
    pattern_library_path: Path
    model_cache_path: Path
    metrics_db_path: Path
    semantic_model_path: Path
    
    @classmethod
    def create(cls, root_dir: str = '.', 
               chroma_dir: str = "data/unified_chroma_db",
               knowledge_graph_path: str = "data/knowledge_graph.json",
               pattern_library_path: str = "data/pattern_library.json",
               model_cache_path: str = "data/model_cache.json",
               metrics_db_path: str = "data/metrics_db.json",
               semantic_model_path: str = "data/semantic_model.json"):
        root = Path(root_dir).resolve()
        return cls(
            root_dir=root,
            chroma_dir=chroma_dir,
            knowledge_graph_path=Path(knowledge_graph_path),
            pattern_library_path=Path(pattern_library_path),
            model_cache_path=Path(model_cache_path),
            metrics_db_path=Path(metrics_db_path),
            semantic_model_path=Path(semantic_model_path)
        )


class KnowledgeGraphManager:
    """Manages the knowledge graph"""
    
    def __init__(self, storage_path: Path):
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path
        self.graph: Dict[str, KnowledgeNode] = {}
        self._load()
    
    def add_node(self, node: KnowledgeNode):
        """Add node to knowledge graph"""
        self.graph[node.node_id] = node
        self.logger.debug(f"Added knowledge node: {node.node_id}")
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get knowledge node by ID"""
        return self.graph.get(node_id)
    
    def connect_nodes(self, node_id_1: str, node_id_2: str):
        """Create bidirectional connection between knowledge nodes"""
        if node_id_1 in self.graph:
            self.graph[node_id_1].connections.add(node_id_2)
        if node_id_2 in self.graph:
            self.graph[node_id_2].connections.add(node_id_1)
        self.logger.debug(f"Connected nodes: {node_id_1} <-> {node_id_2}")
    
    def search(self, query: str, node_type: Optional[str] = None) -> List[KnowledgeNode]:
        """Search knowledge graph by content or type"""
        results = []
        for node in self.graph.values():
            if node_type and node.node_type != node_type:
                continue
            if query.lower() in node.content.lower():
                results.append(node)
        return results
    
    def _load(self):
        """Load knowledge graph from disk"""
        if not self.storage_path.exists():
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for node_id, node_data in data.items():
                    if 'connections' in node_data:
                        node_data['connections'] = set(node_data['connections'])
                    if 'created_at' in node_data:
                        node_data['created_at'] = datetime.fromisoformat(node_data['created_at'])
                    self.graph[node_id] = KnowledgeNode(**node_data)
            self.logger.info(f"Loaded {len(self.graph)} knowledge nodes")
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")
            self.graph = {}
    
    def save(self):
        """Save knowledge graph to disk"""
        try:
            data = {}
            for node_id, node in self.graph.items():
                node_dict = asdict(node)
                node_dict['connections'] = list(node.connections)
                node_dict['created_at'] = node.created_at.isoformat()
                data[node_id] = node_dict

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.graph)} knowledge nodes")
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {e}")


class PatternLibraryManager:
    """Manages the pattern library"""
    
    def __init__(self, storage_path: Path):
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path
        self.library: Dict[str, Pattern] = {}
        self._load()
    
    def add_pattern(self, pattern: Pattern):
        """Add pattern to library"""
        self.library[pattern.pattern_id] = pattern
        self.logger.debug(f"Added pattern: {pattern.name}")
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID"""
        return self.library.get(pattern_id)
    
    def search(self, category: Optional[str] = None,
               min_success_rate: float = 0.0,
               tags: Optional[List[str]] = None) -> List[Pattern]:
        """Search patterns by criteria"""
        results = []
        for pattern in self.library.values():
            if category and pattern.category != category:
                continue
            if pattern.success_rate < min_success_rate:
                continue
            if tags and not any(tag in pattern.tags for tag in tags):
                continue
            results.append(pattern)

        results.sort(key=lambda p: p.success_rate, reverse=True)
        return results
    
    def update_outcome(self, pattern_id: str, success: bool, improvement: float = 0.0):
        """Update pattern outcome statistics"""
        if pattern_id not in self.library:
            return
            
        pattern = self.library[pattern_id]
        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1

        total = pattern.success_count + pattern.failure_count
        pattern.avg_improvement = ((pattern.avg_improvement * (total - 1)) + improvement) / total

        self.logger.debug(f"Updated pattern {pattern.name}: success_rate={pattern.success_rate:.2f}")
    
    def _load(self):
        """Load pattern library from disk"""
        if not self.storage_path.exists():
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for pattern_id, pattern_data in data.items():
                    if 'created_at' in pattern_data:
                        pattern_data['created_at'] = datetime.fromisoformat(pattern_data['created_at'])
                    self.library[pattern_id] = Pattern(**pattern_data)
            self.logger.info(f"Loaded {len(self.library)} patterns")
        except Exception as e:
            self.logger.error(f"Error loading pattern library: {e}")
            self.library = {}
    
    def save(self):
        """Save pattern library to disk"""
        try:
            data = {}
            for pattern_id, pattern in self.library.items():
                pattern_dict = asdict(pattern)
                pattern_dict['created_at'] = pattern.created_at.isoformat()
                data[pattern_id] = pattern_dict

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.library)} patterns")
        except Exception as e:
            self.logger.error(f"Error saving pattern library: {e}")


class ModelCacheManager:
    """Manages LLM response caching"""
    
    def __init__(self, storage_path: Path):
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path
        self.cache: Dict[str, Any] = {}
        self._load()
    
    def get_response(self, prompt: str, model: str = "claude-3.5-sonnet") -> Optional[str]:
        """Get cached LLM response if available"""
        cache_key = self._make_key(prompt, model)
        entry = self.cache.get(cache_key)
        if entry:
            self.logger.debug(f"Cache hit for prompt hash: {cache_key[:8]}...")
            return entry['response']
        return None
    
    def cache_response(self, prompt: str, response: str, model: str = "claude-3.5-sonnet"):
        """Cache LLM response"""
        cache_key = self._make_key(prompt, model)
        self.cache[cache_key] = {
            'prompt': prompt,
            'response': response,
            'model': model,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.debug(f"Cached response for prompt hash: {cache_key[:8]}...")
    
    def _make_key(self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model"""
        content = f"{model}::{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load(self):
        """Load model cache from disk"""
        if not self.storage_path.exists():
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                self.cache = json.load(f)
            self.logger.info(f"Loaded {len(self.cache)} cached responses")
        except Exception as e:
            self.logger.error(f"Error loading model cache: {e}")
            self.cache = {}
    
    def save(self):
        """Save model cache to disk"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.cache, f, indent=2)
            self.logger.info(f"Saved {len(self.cache)} cached responses")
        except Exception as e:
            self.logger.error(f"Error saving model cache: {e}")


class MetricsDatabase:
    """Manages centralized metrics tracking"""

    def __init__(self, storage_path: Path):
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path
        self.db: Dict[str, Any] = {}
        self._load()

    def record(self, category: str, name: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric"""
        if category not in self.db:
            self.db[category] = {}
        if name not in self.db[category]:
            self.db[category][name] = []
        self.db[category][name].append({
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })

    def get(self, category: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics by category and optionally by name"""
        if category not in self.db:
            return {}
        if name:
            return self.db[category].get(name, [])
        return self.db[category]

    def get_latest(self, category: str, name: str) -> Optional[Any]:
        """Get the latest value for a metric"""
        metrics = self.get(category, name)
        if metrics:
            return metrics[-1]['value']
        return None

    def _load(self):
        """Load metrics database from disk"""
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, 'r') as f:
                self.db = json.load(f)
            total = sum(len(metrics) for category in self.db.values() for metrics in category.values())
            self.logger.info(f"Loaded metrics database: {total} total metrics")
        except Exception as e:
            self.logger.error(f"Error loading metrics database: {e}")
            self.db = {}

    def save(self):
        """Save metrics database to disk"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.db, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metrics database: {e}")


class SharedResources:
    """
    Unified access to all shared data structures and services.

    Singleton-like aggregator for all managers.
    """

    def __init__(self, root_dir: str = '.'):
        self.logger = logging.getLogger(__name__)
        self.paths = ResourcePaths.create(root_dir=root_dir)
        self.root_dir = self.paths.root_dir

        # Data directory
        data_dir = self.root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Storage managers
        self.knowledge_graph = KnowledgeGraphManager(self.paths.knowledge_graph_path)
        self.pattern_library = PatternLibraryManager(self.paths.pattern_library_path)
        self.model_cache = ModelCacheManager(self.paths.model_cache_path)
        self.metrics_db = MetricsDatabase(self.paths.metrics_db_path)

        # V2 components
        self.dependency_analyzer = DependencyAnalyzer(root_dir=str(self.root_dir))
        self.code_embedder = CodeEmbedder(chroma_dir=self.paths.chroma_dir)
        self.meta_learner = MetaLearner()

        # Consciousness
        if HAS_CURIOSITY_ENGINE:
            self.curiosity_engine = CuriosityEngine()
            self.logger.info("CuriosityEngine initialized for curiosity-driven learning")
        else:
            self.curiosity_engine = None

        # Semantic model & thought traces
        self.semantic_model = SemanticCodeModel()
        self._load_semantic_model()
        self.thought_trace_manager = ThoughtTraceManager(trace_dir="data/thought_traces")

        self.logger.info(f"SharedResources initialized with root: {self.root_dir}")
        self.logger.info(f"Knowledge Graph: {len(self.knowledge_graph.graph)} nodes")
        self.logger.info(f"Pattern Library: {len(self.pattern_library.library)} patterns")
        self.logger.info(f"Model Cache: {len(self.model_cache.cache)} entries")
        self.logger.info(f"Semantic Model: {len(self.semantic_model.component_understanding)} components")
        self.logger.info(f"Thought Trace Manager: Initialized")

    # ========== Delegate Methods ==========
    # These delegate to sub-managers so the rest of the codebase
    # can call self.resources.method() without knowing internals.

    def get_dependencies(self, file_path: str) -> List[str]:
        return self.dependency_analyzer.get_dependencies(file_path)

    def get_reverse_dependencies(self, file_path: str) -> List[str]:
        return self.dependency_analyzer.get_reverse_dependencies(file_path)

    def get_file_criticality(self, file_path: str) -> float:
        return self.dependency_analyzer.calculate_criticality(file_path)

    def get_dependency_graph(self):
        return self.dependency_analyzer.graph

    def search_similar_code(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            return self.code_embedder.search_similar(query=query, n_results=n_results)
        except Exception:
            return []

    def store_improvement(self, file_path: str, original: str, improved: str, metadata: Dict = None):
        try:
            self.code_embedder.store_improvement(
                file_path=file_path, original_code=original,
                improved_code=improved, metadata=metadata or {}
            )
        except Exception as e:
            self.logger.error(f"Failed to store improvement: {e}")

    def search_patterns(self, category: str = None, min_success_rate: float = 0.0) -> List[Pattern]:
        results = []
        for p in self.pattern_library.library.values():
            if category and p.category != category:
                continue
            if p.success_rate >= min_success_rate:
                results.append(p)
        return sorted(results, key=lambda p: p.success_rate, reverse=True)

    def add_pattern(self, pattern: Pattern):
        self.pattern_library.add(pattern)

    def update_pattern_outcome(self, pattern_id: str, success: bool):
        p = self.pattern_library.library.get(pattern_id)
        if p:
            p.usage_count += 1
            alpha = 0.2
            p.success_rate = (1 - alpha) * p.success_rate + alpha * (1.0 if success else 0.0)

    def add_knowledge_node(self, node: KnowledgeNode):
        self.knowledge_graph.add_node(node)

    def record_metric(self, category: str, name: str, value: Any, metadata: Dict = None):
        self.metrics_db.record(category, name, value, metadata)

    def get_metrics(self, category: str, name: str = None) -> Dict[str, Any]:
        return self.metrics_db.get(category, name)

    def get_latest_metric(self, category: str, name: str) -> Optional[Any]:
        return self.metrics_db.get_latest(category, name)

    def get_cached_response(self, key: str) -> Optional[Any]:
        return self.model_cache.cache.get(key)

    def cache_response(self, key: str, response: Any):
        self.model_cache.cache[key] = response

    def get_component_understanding(self, file_path: str):
        return self.semantic_model.component_understanding.get(file_path)

    def analyze_component_semantics(self, file_path: str, code: str):
        return self.semantic_model.analyze_component(file_path, code)

    # ========== Persistence ==========

    def _load_semantic_model(self):
        try:
            if self.paths.semantic_model_path.exists():
                self.semantic_model.load(str(self.paths.semantic_model_path))
            else:
                self.logger.info("No existing semantic model found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading semantic model: {e}")

    def save_all(self):
        """Save all managers to disk"""
        self.knowledge_graph.save()
        self.pattern_library.save()
        self.model_cache.save()
        self.metrics_db.save()
        try:
            self.semantic_model.save(str(self.paths.semantic_model_path))
        except Exception as e:
            self.logger.error(f"Error saving semantic model: {e}")
        self.logger.info("All resources saved")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'knowledge_graph': {'nodes': len(self.knowledge_graph.graph)},
            'pattern_library': {'total_patterns': len(self.pattern_library.library)},
            'model_cache': {'cached_responses': len(self.model_cache.cache)},
        }