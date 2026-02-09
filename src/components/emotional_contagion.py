import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging
from collections import deque

class EmotionalContagion:
    def __init__(self, 
                 influence_threshold: float = 0.3,
                 sync_rate: float = 0.1,
                 max_history: int = 1000):
        """Initialize emotional contagion system."""
        self.influence_threshold = influence_threshold
        self.sync_rate = sync_rate
        self.emotional_history = deque(maxlen=max_history)
        self.influence_patterns = {}
        self.sync_state = {}
        
        # Initialize neural network for influence prediction
        self._build_influence_model()
        
    def _build_influence_model(self):
        """Build neural network for predicting emotional influence."""
        self.influence_model = nn.Sequential(
            nn.Linear(8, 32),  # Input: emotional state (8 dimensions)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)   # Output: predicted influence
        )
        
    def propagate_emotion(self, 
                         source_state: Dict[str, float],
                         target_state: Dict[str, float]) -> Dict[str, float]:
        """Propagate emotional state from source to target."""
        try:
            # Convert states to tensors
            source_tensor = torch.tensor(list(source_state.values()), dtype=torch.float32)
            target_tensor = torch.tensor(list(target_state.values()), dtype=torch.float32)
            
            # Predict influence
            with torch.no_grad():
                influence = self.influence_model(source_tensor)
            
            # Calculate emotional distance
            distance = torch.norm(source_tensor - target_tensor)
            
            # Apply influence if above threshold
            if distance > self.influence_threshold:
                new_state = target_tensor + self.sync_rate * influence
                new_state = torch.clamp(new_state, 0.0, 1.0)
                
                # Convert back to dictionary
                return dict(zip(target_state.keys(), new_state.numpy()))
            
            return target_state
            
        except Exception as e:
            logging.error(f"Error in emotion propagation: {str(e)}")
            return target_state
            
    def synchronize_emotions(self, 
                           states: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Synchronize multiple emotional states."""
        try:
            if not states:
                return states
                
            # Convert states to tensor
            state_tensor = torch.tensor([list(s.values()) for s in states], dtype=torch.float32)
            
            # Calculate mean state
            mean_state = torch.mean(state_tensor, dim=0)
            
            # Apply synchronization
            synced_states = []
            for state in state_tensor:
                new_state = state + self.sync_rate * (mean_state - state)
                new_state = torch.clamp(new_state, 0.0, 1.0)
                synced_states.append(dict(zip(states[0].keys(), new_state.numpy())))
                
            return synced_states
            
        except Exception as e:
            logging.error(f"Error in emotion synchronization: {str(e)}")
            return states
            
    def update_influence_patterns(self, 
                                source_state: Dict[str, float],
                                target_state: Dict[str, float],
                                success: bool):
        """Update influence patterns based on interaction success."""
        try:
            pattern_key = tuple(source_state.values())
            if pattern_key not in self.influence_patterns:
                self.influence_patterns[pattern_key] = {
                    'success_count': 0,
                    'total_count': 0
                }
                
            self.influence_patterns[pattern_key]['total_count'] += 1
            if success:
                self.influence_patterns[pattern_key]['success_count'] += 1
                
        except Exception as e:
            logging.error(f"Error updating influence patterns: {str(e)}")
            
    def get_influence_stats(self) -> Dict[str, float]:
        """Get statistics about emotional influence patterns."""
        try:
            stats = {
                'total_patterns': len(self.influence_patterns),
                'avg_success_rate': 0.0,
                'most_successful_pattern': None,
                'highest_success_rate': 0.0
            }
            
            if self.influence_patterns:
                success_rates = []
                for pattern, data in self.influence_patterns.items():
                    rate = data['success_count'] / data['total_count']
                    success_rates.append(rate)
                    if rate > stats['highest_success_rate']:
                        stats['highest_success_rate'] = rate
                        stats['most_successful_pattern'] = pattern
                        
                stats['avg_success_rate'] = np.mean(success_rates)
                
            return stats
            
        except Exception as e:
            logging.error(f"Error getting influence stats: {str(e)}")
            return {}
            
    def save_state(self, filepath: str):
        """Save emotional contagion state."""
        try:
            state = {
                'influence_patterns': self.influence_patterns,
                'sync_state': self.sync_state,
                'model_state': self.influence_model.state_dict()
            }
            torch.save(state, filepath)
        except Exception as e:
            logging.error(f"Error saving emotional contagion state: {str(e)}")
            
    def load_state(self, filepath: str):
        """Load emotional contagion state."""
        try:
            state = torch.load(filepath)
            self.influence_patterns = state['influence_patterns']
            self.sync_state = state['sync_state']
            self.influence_model.load_state_dict(state['model_state'])
        except Exception as e:
            logging.error(f"Error loading emotional contagion state: {str(e)}") 