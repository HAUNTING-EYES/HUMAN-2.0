#!/usr/bin/env python3
"""Emotional learning system for HUMAN 2.0."""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
import random
from collections import deque
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """Data class for memory entries."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class InteractionData:
    """Data class for interaction data."""
    emotional_state: np.ndarray
    response_index: int
    next_emotional_state: np.ndarray
    response_appropriateness: float = 0.0
    emotional_stability: float = 0.0
    empathy_effectiveness: float = 0.0
    emotional_intensity: float = 0.0
    personality_consistency: float = 0.0


class NeuralNetworkModel:
    """Handles neural network operations."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float):
        self.model = self._build_model(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _build_model(self, state_size: int, action_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_size)
        )
        
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(state)
            
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                   rewards: torch.Tensor, next_states: torch.Tensor, 
                   dones: torch.Tensor, gamma: float) -> torch.Tensor:
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
        
    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint)


class PatternDetector:
    """Handles pattern detection in emotional history."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.significance_threshold = 0.3
        
    def detect_patterns(self, emotional_history: List[Dict[str, float]]) -> List[Dict]:
        try:
            patterns = []
            if len(emotional_history) < 3:
                return patterns
                
            differences = self._calculate_differences(emotional_history)
            
            for i in range(len(differences) - 2):
                pattern = differences[i:i+3]
                if self._is_pattern_significant(pattern):
                    patterns.append({
                        'pattern': pattern,
                        'confidence': self._calculate_pattern_confidence(pattern),
                        'timestamp': datetime.now().isoformat()
                    })
                    
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return []
            
    def _calculate_differences(self, emotional_history: List[Dict[str, float]]) -> List[Dict[str, float]]:
        differences = []
        for i in range(1, len(emotional_history)):
            diff = {
                k: emotional_history[i][k] - emotional_history[i-1][k]
                for k in emotional_history[i].keys()
                if k != 'timestamp'
            }
            differences.append(diff)
        return differences
        
    def _is_pattern_significant(self, pattern: List[Dict[str, float]]) -> bool:
        try:
            magnitudes = [np.sqrt(sum(v * v for v in diff.values())) for diff in pattern]
            avg_magnitude = np.mean(magnitudes)
            return avg_magnitude > self.significance_threshold
        except Exception as e:
            self.logger.error(f"Error checking pattern significance: {str(e)}")
            return False
            
    def _calculate_pattern_confidence(self, pattern: List[Dict[str, float]]) -> float:
        return 1.0
        
    def calculate_pattern_similarity(self, pattern1: List[Dict[str, float]], 
                                    pattern2: List[Dict[str, float]]) -> float:
        try:
            if len(pattern1) != len(pattern2):
                return 0.0
                
            similarities = []
            for p1, p2 in zip(pattern1, pattern2):
                v1 = np.array(list(p1.values()))
                v2 = np.array(list(p2.values()))
                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                similarities.append(similarity)
                
            return np.mean(similarities)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {str(e)}")
            return 0.0


class RewardCalculator:
    """Calculates rewards for emotional interactions."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def calculate_reward(self, interaction_data: Dict[str, Any]) -> float:
        try:
            reward = 0.0
            
            if interaction_data.get('response_appropriateness', 0) > 0.7:
                reward += 1.0
                
            if interaction_data.get('emotional_stability', 0) > 0.6:
                reward += 0.5
                
            if interaction_data.get('empathy_effectiveness', 0) > 0.8:
                reward += 1.0
                
            if interaction_data.get('emotional_intensity', 0) > 0.9:
                reward -= 0.5
                
            if interaction_data.get('personality_consistency', 0) > 0.7:
                reward += 0.5
                
            return max(-1.0, min(1.0, reward))
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0


class StrategyManager:
    """Manages emotional response strategies."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.strategies = {
            'defensive': {'response_modifier': 0.5},
            'aggressive': {'response_modifier': 1.5},
            'balanced': {'response_modifier': 1.0}
        }
        self.current_strategy = 'balanced'
        
    def adapt_strategy(self, emotional_state: Dict[str, float], 
                      interaction_history: List[Dict]) -> str:
        try:
            stability = self._calculate_emotional_stability(emotional_state)
            success_rate = self._calculate_success_rate(interaction_history)
            
            if stability < 0.3 and success_rate < 0.4:
                self.current_strategy = 'defensive'
            elif stability > 0.7 and success_rate > 0.6:
                self.current_strategy = 'aggressive'
            else:
                self.current_strategy = 'balanced'
                
            return self.current_strategy
            
        except Exception as e:
            self.logger.error(f"Error adapting strategy: {str(e)}")
            return 'balanced'
            
    def _calculate_emotional_stability(self, emotional_state: Dict[str, float]) -> float:
        try:
            values = [v for k, v in emotional_state.items() if k != 'timestamp']
            variance = np.var(values)
            stability = 1.0 / (1.0 + np.exp(variance * 10 - 2))
            return stability
        except Exception as e:
            self.logger.error(f"Error calculating emotional stability: {str(e)}")
            return 0.5
            
    def _calculate_success_rate(self, interaction_history: List[Dict]) -> float:
        try:
            if not interaction_history:
                return 0.5
            successful = sum(1 for interaction in interaction_history 
                           if interaction.get('sentiment', 0) > 0)
            return successful / len(interaction_history)
        except Exception as e:
            self.logger.error(f"Error calculating success rate: {str(e)}")
            return 0.5


class ResponseGenerator:
    """Generates emotional responses."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.templates = {
            'joy': [
                "I'm really happy to hear that!",
                "That's wonderful news!",
                "I'm delighted to hear about this!"
            ],
            'sadness': [
                "I understand this is difficult.",
                "I'm here to support you.",
                "I hear your pain."
            ],
            'anger': [
                "I understand your frustration.",
                "That's definitely concerning.",
                "I can see why you're upset."
            ],
            'fear': [
                "I understand your concern.",
                "Let's work through this together.",
                "I'm here to help you feel safe."
            ],
            'surprise': [
                "That's quite unexpected!",
                "I didn't see that coming.",
                "Wow, that's surprising!"
            ],
            'disgust': [
                "I understand your reaction.",
                "That's quite concerning.",
                "I can see why you feel that way."
            ],
            'neutral': [
                "I understand.",
                "I see.",
                "I hear you."
            ],
            'complex': [
                "I understand the complexity of your feelings.",
                "This situation has many layers.",
                "I appreciate the depth of your emotions."
            ]
        }
        
    def generate_response(self, strategy: Dict[str, float], context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            text = self._generate_response_text(strategy, context)
            return {
                'text': text,
                'style': {
                    'expressiveness': max(strategy.values()),
                    'empathy': strategy.get('complex', 0.0)
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'text': "I understand.",
                'style': {'expressiveness': 0.5, 'empathy': 0.5}
            }
            
    def _generate_response_text(self, strategy: Dict[str, float], context: Dict[str, Any]) -> str:
        try:
            dominant_emotion = max(strategy.items(), key=lambda x: x[1])[0]
            
            if context.get('interaction_type') == 'positive':
                return random.choice(self.templates['joy'])
            elif context.get('interaction_type') == 'negative':
                return random.choice(self.templates['sadness'])
            else:
                return random.choice(self.templates[dominant_emotion])
                
        except Exception as e:
            self.logger.error(f"Error generating response text: {str(e)}")
            return "I understand."


class StateManager:
    """Manages learning state persistence."""
    
    def __init__(self, base_dir: Path, logger: logging.Logger):
        self.base_dir = base_dir
        self.logger = logger
        
    def save_state(self, model: NeuralNetworkModel, optimizer: optim.Optimizer,
                   epsilon: float, patterns: List, weights: Dict,
                   history: List, pattern_memory: List, q_table: Dict,
                   current_strategy: str, path: str) -> None:
        try:
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'emotional_patterns': patterns,
                'pattern_weights': weights,
                'learning_history': history,
                'pattern_memory': pattern_memory,
                'q_table': q_table,
                'current_strategy': current_strategy
            }, path)
        except Exception as e:
            self.logger.error(f"Error saving learning state: {str(e)}")
            
    def load_state(self, model: NeuralNetworkModel, optimizer: optim.Optimizer, path: str) -> Dict[str, Any]:
        try:
            checkpoint = torch.load(path)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint
        except Exception as e:
            self.logger.error(f"Error loading learning state: {str(e)}")
            return {}
            
    def save_patterns(self, patterns: List, weights: Dict, pattern_memory: List, threshold: float) -> None:
        try:
            data_dir = self.base_dir / 'learning'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            patterns_data = {
                'emotional_patterns': patterns,
                'pattern_weights': weights,
                'pattern_memory': pattern_memory,
                'pattern_threshold': threshold
            }
            
            patterns_path = data_dir / 'patterns.json'
            with open(patterns_path, 'w') as f:
                json.dump(patterns_data, f)
            
        except Exception as e:
            self.logger.error(f"Error saving patterns: {str(e)}")
            raise


class EmotionalLearningSystem:
    """A reinforcement learning system for emotional responses that learns from interactions
    and adapts its emotional strategies over time."""
    
    def __init__(self, state_size: int = 768, action_size: int = 8, learning_rate: float = 0.001, base_dir: Optional[Path] = None):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir if base_dir is not None else Path(__file__).resolve().parent
        
        self.nn_model = NeuralNetworkModel(state_size, action_size, learning_rate)
        self.pattern_detector = PatternDetector(self.logger)
        self.reward_calculator = RewardCalculator(self.logger)
        self.strategy_manager = StrategyManager(self.logger)
        self.response_generator = Response