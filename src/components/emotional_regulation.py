import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import deque
from datetime import datetime
import math

class EmotionalRegulation:
    """Emotional regulation system for balancing emotional states."""
    
    def __init__(self, balance_threshold: float = 0.5):
        """Initialize emotional regulation system."""
        self.balance_threshold = balance_threshold
        
        # Initialize emotional dimensions
        self.emotion_dimensions = {
            'valence': 0.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0
        }
        
        # Initialize current state
        self.current_state = self.emotion_dimensions.copy()
        
        # Initialize regulation patterns as dictionary
        self.regulation_patterns = {}
        
        # Initialize emotional history
        self.emotional_history = []
        
        # Initialize regulation stats
        self.regulation_stats = {
            'total_patterns': 0,
            'average_regulation': 0.0,
            'regulation_strength': 0.0,
            'avg_success_rate': 0.0,
            'highest_success_rate': 0.0,
            'history_length': 0,
            'most_successful_pattern': None
        }
        
        # Initialize neural network for regulation
        self._build_regulation_model()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def _build_regulation_model(self):
        """Build neural network for emotional regulation."""
        self.regulation_model = nn.Sequential(
            nn.Linear(8, 32),  # Input: emotional state (8 dimensions)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)   # Output: regulation adjustment
        )
        self.optimizer = torch.optim.Adam(self.regulation_model.parameters(), lr=0.01)
        
    def regulate_emotion(self, current_state: Dict[str, float], 
                        target_state: Dict[str, float]) -> Dict[str, float]:
        """Regulate emotional state towards target state."""
        try:
            regulated_state = {}
            
            for dim in self.emotion_dimensions:
                if dim in current_state and dim in target_state:
                    current = current_state[dim]
                    target = target_state[dim]
                    
                    # Calculate regulation adjustment
                    diff = target - current
                    adjustment = diff * self.balance_threshold
                    
                    # Apply regulation
                    regulated_state[dim] = current + adjustment
                    
            # Update regulation patterns
            self.update_regulation_patterns(current_state, regulated_state, True)
            
            # Add to emotional history
            self.emotional_history.append({
                'timestamp': datetime.now().isoformat(),
                'original': current_state,
                'regulated': regulated_state
            })
            
            return regulated_state
            
        except Exception as e:
            self.logger.error(f"Error regulating emotion: {str(e)}")
            return current_state
            
    def balance_emotions(self, states: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Balance multiple emotional states."""
        try:
            balanced_states = []
            
            for state in states:
                # Calculate average state
                avg_state = self._calculate_average_state(states)
                
                # Balance current state towards average
                balanced = self.regulate_emotion(state, avg_state)
                balanced_states.append(balanced)
                
            return balanced_states
            
        except Exception as e:
            self.logger.error(f"Error balancing emotions: {str(e)}")
            return states
            
    def get_regulation_stats(self) -> Dict[str, Any]:
        """Get current regulation statistics."""
        # Update most successful pattern
        if self.regulation_patterns:
            successful_patterns = {k: v for k, v in self.regulation_patterns.items() if v['success']}
            if successful_patterns:
                self.regulation_stats['most_successful_pattern'] = max(
                    successful_patterns.values(),
                    key=lambda p: p['effectiveness']
                )
                
                # Calculate highest success rate
                highest_rate = max(
                    p['success_count'] / p['total_count']
                    for p in successful_patterns.values()
                )
                self.regulation_stats['highest_success_rate'] = highest_rate
        
        return {
            'total_patterns': len(self.regulation_patterns),
            'average_regulation': self._calculate_average_regulation(),
            'regulation_strength': self._calculate_regulation_strength(),
            'avg_success_rate': self._calculate_avg_success_rate(),
            'highest_success_rate': self.regulation_stats['highest_success_rate'],
            'history_length': len(self.emotional_history),
            'most_successful_pattern': self.regulation_stats['most_successful_pattern'],
            'current_state': self.current_state
        }
        
    def save_state(self, filepath: str) -> None:
        """Save regulation state to file."""
        try:
            state = {
                'current_state': self.current_state,
                'regulation_patterns': self.regulation_patterns,
                'regulation_stats': self.regulation_stats,
                'emotional_history': self.emotional_history,
                'model_state': self.regulation_model.state_dict()
            }
            
            torch.save(state, filepath)
            self.logger.info(f"Regulation state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving regulation state: {str(e)}")
            
    def load_state(self, filepath: str) -> None:
        """Load regulation state from file."""
        try:
            state = torch.load(filepath)
            
            self.current_state = state['current_state']
            self.regulation_patterns = state['regulation_patterns']
            self.regulation_stats = state['regulation_stats']
            self.emotional_history = state['emotional_history']
            self.regulation_model.load_state_dict(state['model_state'])
            
            self.logger.info(f"Regulation state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading regulation state: {str(e)}")
            
    def update_regulation_patterns(self, current_state: Dict[str, float], 
                                 target_state: Dict[str, float], 
                                 success: bool) -> None:
        """Update regulation patterns based on experience."""
        try:
            pattern_key = tuple(current_state.values())
            
            if pattern_key not in self.regulation_patterns:
                self.regulation_patterns[pattern_key] = {
                    'current_state': current_state,
                    'target_state': target_state,
                    'success': success,
                    'success_count': 1 if success else 0,
                    'failure_count': 0 if success else 1,
                    'total_count': 1,
                    'effectiveness': self._calculate_effectiveness(current_state, target_state),
                    'strength': self._calculate_regulation_strength(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                pattern = self.regulation_patterns[pattern_key]
                pattern['success_count'] += 1 if success else 0
                pattern['failure_count'] += 0 if success else 1
                pattern['total_count'] += 1
                pattern['success'] = pattern['success_count'] > pattern['failure_count']
                pattern['effectiveness'] = self._calculate_effectiveness(current_state, target_state)
                pattern['strength'] = self._calculate_regulation_strength()
                pattern['timestamp'] = datetime.now().isoformat()
            
            # Update regulation stats
            self._update_regulation_stats()
            
        except Exception as e:
            self.logger.error(f"Error updating regulation patterns: {str(e)}")
            
    def _update_regulation_stats(self) -> None:
        """Update regulation statistics."""
        self.regulation_stats['total_patterns'] = len(self.regulation_patterns)
        self.regulation_stats['average_regulation'] = self._calculate_average_regulation()
        self.regulation_stats['regulation_strength'] = self._calculate_regulation_strength()
        self.regulation_stats['avg_success_rate'] = self._calculate_avg_success_rate()
        self.regulation_stats['history_length'] = len(self.emotional_history)
        
    def _calculate_average_regulation(self) -> float:
        """Calculate average regulation effectiveness."""
        if not self.regulation_patterns:
            return 0.0
            
        total_effectiveness = sum(p['effectiveness'] for p in self.regulation_patterns.values())
        return total_effectiveness / len(self.regulation_patterns)
        
    def _calculate_regulation_strength(self) -> float:
        """Calculate overall regulation strength."""
        if not self.regulation_patterns:
            return 0.0
            
        total_strength = 0.0
        for pattern in self.regulation_patterns.values():
            if pattern['success']:
                # Calculate distance between current and target states
                distance = 0.0
                for dim in self.emotion_dimensions:
                    if dim in pattern['current_state'] and dim in pattern['target_state']:
                        distance += (pattern['current_state'][dim] - pattern['target_state'][dim]) ** 2
                total_strength += math.sqrt(distance)
                
        return total_strength / len(self.regulation_patterns)
        
    def _calculate_avg_success_rate(self) -> float:
        """Calculate average success rate over time."""
        if not self.regulation_patterns:
            return 0.0
            
        # Consider only recent patterns (last 10)
        recent_patterns = list(self.regulation_patterns.values())[-10:]
        successful_patterns = sum(1 for p in recent_patterns if p['success'])
        return successful_patterns / len(recent_patterns) if recent_patterns else 0.0
        
    def _calculate_effectiveness(self, current_state: Dict[str, float], 
                               target_state: Dict[str, float]) -> float:
        """Calculate regulation effectiveness."""
        total_diff = 0.0
        count = 0
        
        for dim in self.emotion_dimensions:
            if dim in current_state and dim in target_state:
                diff = abs(target_state[dim] - current_state[dim])
                total_diff += diff
                count += 1
                
        if count == 0:
            return 0.0
            
        return 1.0 - (total_diff / count)
        
    def _calculate_average_state(self, states: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average emotional state."""
        avg_state = {}
        
        for dim in self.emotion_dimensions:
            values = [state.get(dim, 0.0) for state in states if dim in state]
            if values:
                avg_state[dim] = sum(values) / len(values)
                
        return avg_state
        
    def get_current_state(self) -> Dict[str, float]:
        """Get current emotional state."""
        return self.current_state.copy()
        
    def update_state(self, new_state: Dict[str, float]) -> None:
        """Update current emotional state."""
        self.current_state = new_state.copy() 