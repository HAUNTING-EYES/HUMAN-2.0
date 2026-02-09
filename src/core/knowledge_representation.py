import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

@dataclass
class Concept:
    id: str
    name: str
    vector: np.ndarray  # Semantic vector representation
    type: str  # concept, entity, relation, etc.
    attributes: Dict[str, Any]
    confidence: float
    source: str  # Where this knowledge came from
    timestamp: float  # When it was added/updated

@dataclass
class Relation:
    id: str
    source_id: str
    target_id: str
    type: str
    vector: np.ndarray  # Relation embedding
    attributes: Dict[str, Any]
    confidence: float
    bidirectional: bool = False

class SemanticEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vector_size = self.model.config.hidden_size
        
    def encode(self, text: str) -> np.ndarray:
        """Encode text into semantic vector"""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use CLS token embedding as semantic vector
        vector = outputs.last_hidden_state[:, 0, :].numpy()
        return vector[0]  # Return first (only) vector

class KnowledgeGraph:
    def __init__(self, vector_size: int = 384):
        # Core components
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, Relation] = {}
        self.graph = nx.MultiDiGraph()
        
        # Semantic encoding
        self.encoder = SemanticEncoder()
        self.vector_size = self.encoder.vector_size
        
        # Indexing structures
        self.type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> concept_ids
        self.attribute_index: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))  # attr -> value -> concept_ids
        
        # Analysis components
        self.pca = PCA(n_components=3)  # For dimensionality reduction in analysis
        
    def add_concept(self, 
                   name: str,
                   concept_type: str,
                   attributes: Dict[str, Any],
                   source: str,
                   confidence: float,
                   vector: Optional[np.ndarray] = None) -> str:
        """Add new concept to knowledge graph"""
        # Generate vector if not provided
        if vector is None:
            vector = self.encoder.encode(name)
            
        # Create concept
        concept_id = f"c_{len(self.concepts)}"
        concept = Concept(
            id=concept_id,
            name=name,
            vector=vector,
            type=concept_type,
            attributes=attributes,
            confidence=confidence,
            source=source,
            timestamp=time.time()
        )
        
        # Add to storage
        self.concepts[concept_id] = concept
        self.graph.add_node(concept_id, **concept.__dict__)
        
        # Update indices
        self.type_index[concept_type].add(concept_id)
        for attr, value in attributes.items():
            self.attribute_index[attr][value].add(concept_id)
            
        return concept_id
        
    def add_relation(self,
                    source_id: str,
                    target_id: str,
                    relation_type: str,
                    attributes: Dict[str, Any],
                    confidence: float,
                    bidirectional: bool = False) -> str:
        """Add relation between concepts"""
        # Verify concepts exist
        if source_id not in self.concepts or target_id not in self.concepts:
            raise ValueError("Source or target concept does not exist")
            
        # Generate relation vector
        source_vec = self.concepts[source_id].vector
        target_vec = self.concepts[target_id].vector
        relation_vec = (source_vec + target_vec) / 2  # Simple average for now
        
        # Create relation
        relation_id = f"r_{len(self.relations)}"
        relation = Relation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            type=relation_type,
            vector=relation_vec,
            attributes=attributes,
            confidence=confidence,
            bidirectional=bidirectional
        )
        
        # Add to storage
        self.relations[relation_id] = relation
        self.graph.add_edge(
            source_id,
            target_id,
            key=relation_id,
            **relation.__dict__
        )
        
        if bidirectional:
            self.graph.add_edge(
                target_id,
                source_id,
                key=f"{relation_id}_rev",
                **relation.__dict__
            )
            
        return relation_id
        
    def get_similar_concepts(self, 
                           concept_id: str, 
                           n: int = 5,
                           min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """Find similar concepts using vector similarity"""
        if concept_id not in self.concepts:
            raise ValueError("Concept does not exist")
            
        concept = self.concepts[concept_id]
        similarities = []
        
        for other_id, other in self.concepts.items():
            if other_id != concept_id:
                similarity = 1 - cosine(concept.vector, other.vector)
                if similarity >= min_similarity:
                    similarities.append((other_id, similarity))
                    
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
        
    def find_path(self, 
                 source_id: str, 
                 target_id: str, 
                 max_length: int = 5) -> List[str]:
        """Find shortest path between concepts"""
        try:
            path = nx.shortest_path(
                self.graph,
                source=source_id,
                target=target_id,
                weight=lambda u, v, d: 1 - d['confidence']
            )
            if len(path) > max_length:
                return []
            return path
        except nx.NetworkXNoPath:
            return []
            
    def get_subgraph(self, 
                     concept_ids: List[str], 
                     depth: int = 1) -> nx.MultiDiGraph:
        """Get subgraph centered on given concepts"""
        # Start with initial concepts
        nodes = set(concept_ids)
        
        # Add neighbors up to depth
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.predecessors(node))
                new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)
            
        return self.graph.subgraph(nodes)
        
    def query_concepts(self,
                      concept_type: Optional[str] = None,
                      attributes: Optional[Dict[str, Any]] = None,
                      min_confidence: float = 0.0) -> List[str]:
        """Query concepts based on type and attributes"""
        # Start with all concepts or concepts of specific type
        if concept_type:
            concept_ids = self.type_index[concept_type]
        else:
            concept_ids = set(self.concepts.keys())
            
        # Filter by attributes
        if attributes:
            for attr, value in attributes.items():
                concept_ids &= self.attribute_index[attr][value]
                
        # Filter by confidence
        concept_ids = {
            cid for cid in concept_ids
            if self.concepts[cid].confidence >= min_confidence
        }
        
        return list(concept_ids)
        
    def analyze_concept_clusters(self, min_concepts: int = 10) -> Dict[str, List[str]]:
        """Analyze concept clustering in vector space"""
        if len(self.concepts) < min_concepts:
            return {}
            
        # Get all concept vectors
        vectors = np.array([c.vector for c in self.concepts.values()])
        concept_ids = list(self.concepts.keys())
        
        # Reduce dimensionality for clustering
        vectors_reduced = self.pca.fit_transform(vectors)
        
        # Perform clustering (using simple k-means for now)
        from sklearn.cluster import KMeans
        n_clusters = min(len(concept_ids) // 5, 10)  # Heuristic for number of clusters
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(vectors_reduced)
        
        # Organize results
        cluster_map = defaultdict(list)
        for concept_id, cluster_id in zip(concept_ids, clusters):
            cluster_map[f"cluster_{cluster_id}"].append(concept_id)
            
        return dict(cluster_map)
        
    def get_concept_importance(self, concept_id: str) -> float:
        """Calculate concept importance using PageRank"""
        pagerank = nx.pagerank(self.graph, weight='confidence')
        return pagerank.get(concept_id, 0.0)
        
    def merge_concepts(self, concept_id1: str, concept_id2: str) -> str:
        """Merge two concepts that represent the same thing"""
        c1 = self.concepts[concept_id1]
        c2 = self.concepts[concept_id2]
        
        # Create merged concept
        merged_vector = (c1.vector + c2.vector) / 2
        merged_attributes = {**c1.attributes, **c2.attributes}
        merged_confidence = max(c1.confidence, c2.confidence)
        
        # Create new concept
        merged_id = self.add_concept(
            name=f"{c1.name}|{c2.name}",
            concept_type=c1.type,
            attributes=merged_attributes,
            source=f"merged:{c1.source}+{c2.source}",
            confidence=merged_confidence,
            vector=merged_vector
        )
        
        # Update relations
        for relation in self.relations.values():
            if relation.source_id in (concept_id1, concept_id2):
                self.add_relation(
                    merged_id,
                    relation.target_id,
                    relation.type,
                    relation.attributes,
                    relation.confidence,
                    relation.bidirectional
                )
            if relation.target_id in (concept_id1, concept_id2):
                self.add_relation(
                    relation.source_id,
                    merged_id,
                    relation.type,
                    relation.attributes,
                    relation.confidence,
                    relation.bidirectional
                )
                
        # Remove old concepts
        self.remove_concept(concept_id1)
        self.remove_concept(concept_id2)
        
        return merged_id
        
    def remove_concept(self, concept_id: str):
        """Remove concept and its relations"""
        if concept_id not in self.concepts:
            return
            
        # Remove from indices
        concept = self.concepts[concept_id]
        self.type_index[concept.type].remove(concept_id)
        for attr, value in concept.attributes.items():
            self.attribute_index[attr][value].remove(concept_id)
            
        # Remove relations
        relations_to_remove = []
        for relation_id, relation in self.relations.items():
            if relation.source_id == concept_id or relation.target_id == concept_id:
                relations_to_remove.append(relation_id)
                
        for relation_id in relations_to_remove:
            del self.relations[relation_id]
            
        # Remove from graph
        self.graph.remove_node(concept_id)
        
        # Remove from concepts
        del self.concepts[concept_id]
        
    def save(self, filepath: str):
        """Save knowledge graph to file"""
        import pickle
        data = {
            'concepts': self.concepts,
            'relations': self.relations,
            'type_index': self.type_index,
            'attribute_index': dict(self.attribute_index)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeGraph':
        """Load knowledge graph from file"""
        import pickle
        kg = cls()
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        kg.concepts = data['concepts']
        kg.relations = data['relations']
        kg.type_index = data['type_index']
        kg.attribute_index = defaultdict(lambda: defaultdict(set))
        kg.attribute_index.update(data['attribute_index'])
        
        # Rebuild graph
        for concept in kg.concepts.values():
            kg.graph.add_node(concept.id, **concept.__dict__)
            
        for relation in kg.relations.values():
            kg.graph.add_edge(
                relation.source_id,
                relation.target_id,
                key=relation.id,
                **relation.__dict__
            )
            if relation.bidirectional:
                kg.graph.add_edge(
                    relation.target_id,
                    relation.source_id,
                    key=f"{relation.id}_rev",
                    **relation.__dict__
                )
                
        return kg 