import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

class ContinuousLearningSystem:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
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