import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import logging
from collections import deque
import random
from datetime import datetime
import math

class EmotionalAdaptation:
    """Emotional adaptation system for evolving emotional strategies."""
    
    def __init__(self, adaptation_rate: float = 0.1):
        """Initialize emotional adaptation system."""
        self.adaptation_rate = adaptation_rate
        
        # Initialize emotional dimensions
        self.emotion_dimensions = {
            'valence': 0.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'novelty': 0.0,
            'complexity': 0.0,
            'intensity': 0.0,
            'stability': 0.0,
            'coherence': 0.0
        }
        
        # Initialize current state
        self.current_state = self.emotion_dimensions.copy()
        
        # Initialize adaptation patterns
        self.adaptation_patterns = []
        self.emotional_history = []
        
        # Initialize adaptation stats
        self.adaptation_stats = {
            'total_patterns': 0,
            'success_rate': 0.0,
            'adaptation_strength': 0.0,
            'avg_success_rate': 0.0
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def evolve_strategy(self, current_strategy: Dict[str, float], 
                       performance: float) -> Dict[str, float]:
        """Evolve emotional strategy based on performance."""
        try:
            # Calculate adaptation factor
            adaptation_factor = self.adaptation_rate * performance
            
            # Evolve each dimension
            evolved_strategy = {}
            for dim in self.emotion_dimensions:
                if dim in current_strategy:
                    # Add noise for exploration
                    noise = random.uniform(-0.1, 0.1)
                    evolved_value = current_strategy[dim] + (adaptation_factor + noise)
                    
                    # Ensure value stays in [0,1] range
                    evolved_strategy[dim] = max(0.0, min(1.0, evolved_value))
                    
            return evolved_strategy
            
        except Exception as e:
            self.logger.error(f"Error evolving strategy: {str(e)}")
            return current_strategy
            
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get current adaptation statistics."""
        return {
            'total_patterns': len(self.adaptation_patterns),
            'success_rate': self._calculate_success_rate(),
            'adaptation_strength': self._calculate_adaptation_strength(),
            'avg_success_rate': self._calculate_avg_success_rate(),
            'current_state': self.current_state
        }
        
    def save_state(self, filepath: str) -> None:
        """Save adaptation state to file."""
        try:
            state = {
                'current_state': self.current_state,
                'adaptation_patterns': self.adaptation_patterns,
                'emotional_history': self.emotional_history,
                'adaptation_stats': self.adaptation_stats
            }
            
            torch.save(state, filepath)
            self.logger.info(f"Adaptation state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving adaptation state: {str(e)}")
            
    def load_state(self, filepath: str) -> None:
        """Load adaptation state from file."""
        try:
            state = torch.load(filepath)
            
            self.current_state = state['current_state']
            self.adaptation_patterns = state['adaptation_patterns']
            self.emotional_history = state['emotional_history']
            self.adaptation_stats = state['adaptation_stats']
            
            self.logger.info(f"Adaptation state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading adaptation state: {str(e)}")
            
    def update_adaptation_patterns(self, current_state: Dict[str, float], 
                                 target_state: Dict[str, float], 
                                 success: bool) -> None:
        """Update adaptation patterns based on experience."""
        try:
            pattern = {
                'current_state': current_state,
                'target_state': target_state,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            self.adaptation_patterns.append(pattern)
            
            # Update emotional history
            self.emotional_history.append({
                'current_state': current_state,
                'target_state': target_state,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update adaptation stats
            self._update_adaptation_stats()
            
        except Exception as e:
            self.logger.error(f"Error updating adaptation patterns: {str(e)}")
            
    def _update_adaptation_stats(self) -> None:
        """Update adaptation statistics."""
        self.adaptation_stats['total_patterns'] = len(self.adaptation_patterns)
        self.adaptation_stats['success_rate'] = self._calculate_success_rate()
        self.adaptation_stats['adaptation_strength'] = self._calculate_adaptation_strength()
        self.adaptation_stats['avg_success_rate'] = self._calculate_avg_success_rate()
        
    def _calculate_success_rate(self) -> float:
        """Calculate current success rate."""
        if not self.adaptation_patterns:
            return 0.0
            
        successful_patterns = sum(1 for p in self.adaptation_patterns if p['success'])
        return successful_patterns / len(self.adaptation_patterns)
        
    def _calculate_adaptation_strength(self) -> float:
        """Calculate overall adaptation strength."""
        if not self.adaptation_patterns:
            return 0.0
            
        total_strength = 0.0
        for pattern in self.adaptation_patterns:
            if pattern['success']:
                # Calculate distance between current and target states
                distance = 0.0
                for dim in self.emotion_dimensions:
                    if dim in pattern['current_state'] and dim in pattern['target_state']:
                        distance += (pattern['current_state'][dim] - pattern['target_state'][dim]) ** 2
                total_strength += math.sqrt(distance)
                
        return total_strength / len(self.adaptation_patterns)
        
    def _calculate_avg_success_rate(self) -> float:
        """Calculate average success rate over time."""
        if not self.adaptation_patterns:
            return 0.0
            
        # Consider only recent patterns (last 10)
        recent_patterns = self.adaptation_patterns[-10:]
        successful_patterns = sum(1 for p in recent_patterns if p['success'])
        return successful_patterns / len(recent_patterns)
        
    def get_current_state(self) -> Dict[str, float]:
        """Get current emotional state."""
        return self.current_state.copy()
        
    def update_state(self, new_state: Dict[str, float]) -> None:
        """Update current emotional state."""
        self.current_state = new_state.copy() 