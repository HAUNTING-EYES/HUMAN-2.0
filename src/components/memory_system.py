import numpy as np
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from collections import deque
import heapq

class EnhancedMemorySystem:
    """
    An enhanced memory system with improved consolidation, retrieval, and importance scoring.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the enhanced memory system.
        
        Args:
            base_dir: Base directory for memory storage
        """
        self.base_dir = Path(base_dir) if base_dir else Path("memory")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory buffers
        self.short_term_buffer = deque(maxlen=1000)
        self.working_memory = deque(maxlen=100)
        self.long_term_memory = []
        
        # Memory importance model
        self.importance_model = self._build_importance_model()
        self.importance_optimizer = torch.optim.Adam(self.importance_model.parameters())
        
        # Memory consolidation parameters
        self.consolidation_threshold = 0.7
        self.consolidation_batch_size = 32
        self.min_importance_score = 0.3
        
        # Load existing memories
        self._load_memories()
        
    def _build_importance_model(self) -> nn.Module:
        """
        Build the neural network for memory importance scoring.
        
        Returns:
            Neural network model
        """
        model = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        return model
        
    def store_memory(self, memory_data: Dict[str, Any]) -> None:
        """
        Store a new memory with importance scoring.
        
        Args:
            memory_data: Dictionary containing memory information
        """
        try:
            # Add metadata
            memory_data['timestamp'] = datetime.now().isoformat()
            memory_data['importance_score'] = self._calculate_importance(memory_data)
            
            # Store in short-term buffer
            self.short_term_buffer.append(memory_data)
            
            # Trigger consolidation if buffer is full
            if len(self.short_term_buffer) >= self.consolidation_batch_size:
                self.consolidate_memories()
                
        except Exception as e:
            logging.error(f"Error storing memory: {str(e)}")
            
    def _calculate_importance(self, memory_data: Dict[str, Any]) -> float:
        """
        Calculate memory importance score using the neural network.
        
        Args:
            memory_data: Dictionary containing memory information
            
        Returns:
            Importance score between 0 and 1
        """
        try:
            # Extract features for importance scoring
            features = self._extract_memory_features(memory_data)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
                importance_score = self.importance_model(features_tensor).item()
                
            return float(importance_score)
            
        except Exception as e:
            logging.error(f"Error calculating importance: {str(e)}")
            return 0.5
            
    def _extract_memory_features(self, memory_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from memory data for importance scoring.
        
        Args:
            memory_data: Dictionary containing memory information
            
        Returns:
            Feature vector
        """
        features = []
        
        # Emotional intensity
        features.append(memory_data.get('emotional_intensity', 0.5))
        
        # Interaction significance
        features.append(memory_data.get('interaction_significance', 0.5))
        
        # Novelty score
        features.append(memory_data.get('novelty_score', 0.5))
        
        # Context relevance
        features.append(memory_data.get('context_relevance', 0.5))
        
        # Emotional valence
        features.append(memory_data.get('emotional_valence', 0.5))
        
        # Social significance
        features.append(memory_data.get('social_significance', 0.5))
        
        # Memory age (normalized)
        age = (datetime.now() - datetime.fromisoformat(memory_data['timestamp'])).total_seconds()
        features.append(min(1.0, age / (24 * 3600)))  # Normalize to 24 hours
        
        # Memory complexity
        features.append(memory_data.get('complexity_score', 0.5))
        
        return np.array(features)
        
    def consolidate_memories(self) -> None:
        """
        Consolidate memories from short-term to long-term storage.
        """
        try:
            # Sort memories by importance
            memories = sorted(self.short_term_buffer, 
                           key=lambda x: x['importance_score'], 
                           reverse=True)
            
            # Keep top memories based on importance
            for memory in memories:
                if memory['importance_score'] >= self.min_importance_score:
                    self.long_term_memory.append(memory)
                    
            # Clear short-term buffer
            self.short_term_buffer.clear()
            
            # Save consolidated memories
            self._save_memories()
            
        except Exception as e:
            logging.error(f"Error consolidating memories: {str(e)}")
            
    def retrieve_memories(self, query: Dict[str, Any], 
                         max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query.
        
        Args:
            query: Dictionary containing retrieval criteria
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant memories
        """
        try:
            relevant_memories = []
            
            # Calculate relevance scores
            for memory in self.long_term_memory:
                relevance_score = self._calculate_relevance(memory, query)
                if relevance_score > 0.5:  # Relevance threshold
                    memory['relevance_score'] = relevance_score
                    relevant_memories.append(memory)
                    
            # Sort by relevance and importance
            relevant_memories.sort(key=lambda x: (x['relevance_score'], x['importance_score']), 
                                reverse=True)
            
            return relevant_memories[:max_results]
            
        except Exception as e:
            logging.error(f"Error retrieving memories: {str(e)}")
            return []
            
    def _calculate_relevance(self, memory: Dict[str, Any], 
                           query: Dict[str, Any]) -> float:
        """
        Calculate relevance score between memory and query.
        
        Args:
            memory: Memory dictionary
            query: Query dictionary
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            relevance_score = 0.0
            weights = {
                'emotional_similarity': 0.3,
                'context_similarity': 0.3,
                'temporal_relevance': 0.2,
                'importance_weight': 0.2
            }
            
            # Emotional similarity
            if 'emotions' in memory and 'emotions' in query:
                emotional_similarity = self._calculate_emotional_similarity(
                    memory['emotions'], query['emotions'])
                relevance_score += weights['emotional_similarity'] * emotional_similarity
                
            # Context similarity
            if 'context' in memory and 'context' in query:
                context_similarity = self._calculate_context_similarity(
                    memory['context'], query['context'])
                relevance_score += weights['context_similarity'] * context_similarity
                
            # Temporal relevance
            memory_time = datetime.fromisoformat(memory['timestamp'])
            query_time = datetime.fromisoformat(query.get('timestamp', datetime.now().isoformat()))
            time_diff = abs((memory_time - query_time).total_seconds())
            temporal_relevance = 1.0 / (1.0 + time_diff / (24 * 3600))  # Decay over 24 hours
            relevance_score += weights['temporal_relevance'] * temporal_relevance
            
            # Importance weight
            relevance_score += weights['importance_weight'] * memory['importance_score']
            
            return min(1.0, max(0.0, relevance_score))
            
        except Exception as e:
            logging.error(f"Error calculating relevance: {str(e)}")
            return 0.0
            
    def _calculate_emotional_similarity(self, emotions1: Dict[str, float], 
                                     emotions2: Dict[str, float]) -> float:
        """
        Calculate similarity between two emotional states.
        
        Args:
            emotions1: First emotional state
            emotions2: Second emotional state
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Convert to vectors
            vec1 = np.array([emotions1.get(e, 0.0) for e in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']])
            vec2 = np.array([emotions2.get(e, 0.0) for e in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']])
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error calculating emotional similarity: {str(e)}")
            return 0.0
            
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                   context2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two contexts.
        
        Args:
            context1: First context
            context2: Second context
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Extract context features
            features1 = self._extract_context_features(context1)
            features2 = self._extract_context_features(context2)
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error calculating context similarity: {str(e)}")
            return 0.0
            
    def _extract_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from context for similarity calculation.
        
        Args:
            context: Context dictionary
            
        Returns:
            Feature vector
        """
        features = []
        
        # Environment type
        features.append(float(context.get('environment_type', 0)))
        
        # Social presence
        features.append(float(context.get('social_presence', 0)))
        
        # Time of day (normalized)
        time = datetime.fromisoformat(context.get('timestamp', datetime.now().isoformat()))
        features.append(time.hour / 24.0)
        
        # Location type
        features.append(float(context.get('location_type', 0)))
        
        # Activity type
        features.append(float(context.get('activity_type', 0)))
        
        return np.array(features)
        
    def _load_memories(self) -> None:
        """
        Load memories from storage.
        """
        try:
            memory_file = self.base_dir / "memories.json"
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    self.long_term_memory = json.load(f)
        except Exception as e:
            logging.error(f"Error loading memories: {str(e)}")
            
    def _save_memories(self) -> None:
        """
        Save memories to storage.
        """
        try:
            memory_file = self.base_dir / "memories.json"
            with open(memory_file, 'w') as f:
                json.dump(self.long_term_memory, f)
        except Exception as e:
            logging.error(f"Error saving memories: {str(e)}")
            
    def prune_memories(self, max_size: int = 10000) -> None:
        """
        Prune memories to maintain size limit.
        
        Args:
            max_size: Maximum number of memories to keep
        """
        try:
            if len(self.long_term_memory) > max_size:
                # Sort by importance and recency
                self.long_term_memory.sort(
                    key=lambda x: (x['importance_score'], 
                                 datetime.fromisoformat(x['timestamp']).timestamp()),
                    reverse=True
                )
                
                # Keep top memories
                self.long_term_memory = self.long_term_memory[:max_size]
                
                # Save pruned memories
                self._save_memories()
                
        except Exception as e:
            logging.error(f"Error pruning memories: {str(e)}")
            
    def update_importance_scores(self) -> None:
        """
        Update importance scores for all memories.
        """
        try:
            for memory in self.long_term_memory:
                memory['importance_score'] = self._calculate_importance(memory)
                
            # Save updated memories
            self._save_memories()
            
        except Exception as e:
            logging.error(f"Error updating importance scores: {str(e)}")
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary containing memory statistics
        """
        try:
            return {
                'short_term_count': len(self.short_term_buffer),
                'long_term_count': len(self.long_term_memory),
                'avg_importance': np.mean([m['importance_score'] for m in self.long_term_memory]),
                'max_importance': max([m['importance_score'] for m in self.long_term_memory]),
                'min_importance': min([m['importance_score'] for m in self.long_term_memory]),
                'memory_age_range': {
                    'oldest': min([datetime.fromisoformat(m['timestamp']) for m in self.long_term_memory]),
                    'newest': max([datetime.fromisoformat(m['timestamp']) for m in self.long_term_memory])
                }
            }
        except Exception as e:
            logging.error(f"Error getting memory stats: {str(e)}")
            return {} 