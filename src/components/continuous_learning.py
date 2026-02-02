import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import torch
import time

class ContinuousLearningSystem:
    """System for continuous learning and improvement."""
    
    def __init__(self, base_dir: str):
        """Initialize the continuous learning system.
        
        Args:
            base_dir: Base directory for storing learning data
        """
        self.base_dir = Path(base_dir)
        self.learning_dir = self.base_dir / "continuous_learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector store
        self.vector_store = chromadb.PersistentClient(path=os.path.join(base_dir, "vector_store"))
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        try:
            self.collection = self.vector_store.get_collection("learning_data")
        except ValueError:
            self.collection = self.vector_store.create_collection("learning_data")
        
        # Load learning history
        self.learning_history = self._load_learning_history()
        
        # Initialize learning state
        self.learning_state = {
            "knowledge_base": {},
            "improvement_history": [],
            "performance_metrics": {}
        }
        
        # Load existing state if available
        self._load_state()
        
    def learn_from_interaction(self, interaction_data: Dict[str, Any]):
        """Learn from a new interaction"""
        try:
            # Extract relevant information
            text = interaction_data.get('text', '')
            code = interaction_data.get('code', '')
            outcome = interaction_data.get('outcome', '')
            
            # Generate embeddings
            text_embedding = self.model.encode(text).tolist()
            code_embedding = self.model.encode(code).tolist()
            
            # Store in vector database
            self.collection.add(
                embeddings=[text_embedding, code_embedding],
                documents=[text, code],
                metadatas=[{
                    'type': 'text',
                    'timestamp': datetime.now().isoformat(),
                    'outcome': outcome
                }, {
                    'type': 'code',
                    'timestamp': datetime.now().isoformat(),
                    'outcome': outcome
                }],
                ids=[f"text_{len(self.learning_history)}", f"code_{len(self.learning_history)}"]
            )
            
            # Update learning history
            self.learning_history.append({
                'timestamp': datetime.now().isoformat(),
                'interaction': interaction_data,
                'outcome': outcome
            })
            
            # Save updated history
            self._save_learning_history()
            
            self.logger.info("Successfully learned from new interaction")
            
        except Exception as e:
            self.logger.error(f"Error learning from interaction: {str(e)}")
            
    def find_similar_patterns(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar patterns in learning history"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            similar_patterns = []
            for i in range(len(results['documents'][0])):
                similar_patterns.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
                
            return similar_patterns
            
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {str(e)}")
            return []
            
    def analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress and effectiveness"""
        analysis = {
            'total_interactions': len(self.learning_history),
            'success_rate': 0,
            'learning_rate': 0,
            'pattern_effectiveness': {},
            'improvement_areas': []
        }
        
        if not self.learning_history:
            return analysis
            
        # Calculate success rate
        successful_outcomes = sum(1 for h in self.learning_history if h['outcome'] == 'success')
        analysis['success_rate'] = (successful_outcomes / len(self.learning_history)) * 100
        
        # Calculate learning rate
        recent_interactions = self.learning_history[-10:]  # Last 10 interactions
        recent_success = sum(1 for h in recent_interactions if h['outcome'] == 'success')
        analysis['learning_rate'] = (recent_success / len(recent_interactions)) * 100
        
        # Analyze pattern effectiveness
        for interaction in self.learning_history:
            pattern_type = interaction['interaction'].get('type', 'unknown')
            if pattern_type not in analysis['pattern_effectiveness']:
                analysis['pattern_effectiveness'][pattern_type] = {
                    'total': 0,
                    'successful': 0
                }
            analysis['pattern_effectiveness'][pattern_type]['total'] += 1
            if interaction['outcome'] == 'success':
                analysis['pattern_effectiveness'][pattern_type]['successful'] += 1
                
        # Identify improvement areas
        for pattern_type, stats in analysis['pattern_effectiveness'].items():
            success_rate = (stats['successful'] / stats['total']) * 100
            if success_rate < 50:
                analysis['improvement_areas'].append({
                    'pattern': pattern_type,
                    'success_rate': success_rate,
                    'suggestion': f"Improve effectiveness of {pattern_type} patterns"
                })
                
        return analysis
        
    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on learning analysis"""
        improvements = []
        analysis = self.analyze_learning_progress()
        
        # Suggest improvements based on success rate
        if analysis['success_rate'] < 70:
            improvements.append({
                'type': 'learning_effectiveness',
                'description': f"Low success rate ({analysis['success_rate']:.1f}%). Consider reviewing and improving learning patterns.",
                'priority': 'high'
            })
            
        # Suggest improvements based on learning rate
        if analysis['learning_rate'] < 60:
            improvements.append({
                'type': 'learning_rate',
                'description': f"Recent learning rate ({analysis['learning_rate']:.1f}%) is below target. May need to adjust learning strategies.",
                'priority': 'medium'
            })
            
        # Suggest improvements for specific patterns
        for area in analysis['improvement_areas']:
            improvements.append({
                'type': 'pattern_improvement',
                'description': area['suggestion'],
                'priority': 'medium',
                'pattern': area['pattern'],
                'current_success_rate': area['success_rate']
            })
            
        return improvements
        
    def _load_learning_history(self) -> List[Dict[str, Any]]:
        """Load learning history from disk"""
        history_file = os.path.join(self.base_dir, 'learning_history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading learning history: {str(e)}")
        return []
        
    def _save_learning_history(self):
        """Save learning history to disk"""
        history_file = os.path.join(self.base_dir, 'learning_history.json')
        try:
            with open(history_file, 'w') as f:
                json.dump(self.learning_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving learning history: {str(e)}")
            
    def learn(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a learning task and update knowledge.
        
        Args:
            task_data: Dictionary containing task data and metrics
            
        Returns:
            Dictionary containing learning results
        """
        try:
            # Extract task information
            task_type = task_data.get("type", "unknown")
            content = task_data.get("content", "")
            context = task_data.get("context", {})
            
            if not content:
                return {
                    "status": "error",
                    "message": "No content provided in task data"
                }
            
            # Update knowledge base
            knowledge_entry = {
                "type": task_type,
                "content": content,
                "context": context,
                "timestamp": task_data.get("timestamp", "")
            }
            
            if task_type not in self.learning_state["knowledge_base"]:
                self.learning_state["knowledge_base"][task_type] = []
            self.learning_state["knowledge_base"][task_type].append(knowledge_entry)
            
            # Record learning attempt
            self.learning_state["improvement_history"].append({
                "task_type": task_type,
                "content": content,
                "timestamp": task_data.get("timestamp", "")
            })
            
            # Save updated state
            self._save_state()
            
            return {
                "status": "success",
                "knowledge_entry": knowledge_entry,
                "knowledge_base_size": len(self.learning_state["knowledge_base"][task_type])
            }
            
        except Exception as e:
            self.logger.error(f"Error in learning process: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
        
    def improve(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a continuous learning task.
        
        Args:
            task_data: Dictionary containing task information and learning data
            
        Returns:
            Dictionary containing learning results and model updates
        """
        try:
            # Extract learning data
            learning_data = task_data.get('learning_data', {})
            current_metrics = task_data.get('metrics', {})
            
            # Update learning history
            self.learning_history.append({
                'timestamp': datetime.now().isoformat(),
                'task': task_data,
                'metrics': current_metrics
            })
            
            # Process new knowledge
            knowledge_updates = self._process_new_knowledge(learning_data)
            
            # Update model if needed
            model_updates = self._update_model(current_metrics)
            
            return {
                'success': True,
                'knowledge_updates': knowledge_updates,
                'model_updates': model_updates,
                'current_metrics': current_metrics,
                'history_length': len(self.learning_history)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def _process_new_knowledge(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and integrate new knowledge.
        
        Args:
            learning_data: Dictionary containing new knowledge to process
            
        Returns:
            Dictionary containing knowledge processing results
        """
        updates = {
            'processed_items': 0,
            'new_concepts': [],
            'updated_concepts': []
        }
        
        for item in learning_data.get('items', []):
            # Process each knowledge item
            concept = item.get('concept')
            if concept:
                if concept not in self.learning_state["knowledge_base"]:
                    updates['new_concepts'].append(concept)
                else:
                    updates['updated_concepts'].append(concept)
                    
                self.learning_state["knowledge_base"][concept] = item
                updates['processed_items'] += 1
                
        return updates
        
    def _update_model(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update the learning model based on performance metrics.
        
        Args:
            metrics: Dictionary of current performance metrics
            
        Returns:
            Dictionary containing model update results
        """
        updates = {
            'learning_rate_adjusted': False,
            'model_complexity_adjusted': False,
            'parameters_updated': False
        }
        
        # Adjust learning rate if needed
        if metrics.get('loss', float('inf')) > self.target_loss:
            self.learning_state["performance_metrics"]["learning_rate"] *= 0.9
            updates['learning_rate_adjusted'] = True
            
        # Update model parameters
        if len(self.learning_history) >= self.batch_size:
            # Perform batch update
            updates['parameters_updated'] = True
            
        return updates
    
    def _load_state(self):
        """Load learning state from file."""
        state_file = self.learning_dir / "learning_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    self.learning_state = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading learning state: {str(e)}")
    
    def _save_state(self):
        """Save learning state to file."""
        state_file = self.learning_dir / "learning_state.json"
        try:
            with open(state_file, "w") as f:
                json.dump(self.learning_state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving learning state: {str(e)}") 