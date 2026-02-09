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

class EmotionalLearningSystem:
    """
    A reinforcement learning system for emotional responses that learns from interactions
    and adapts its emotional strategies over time.
    """
    
    def __init__(self, state_size: int = 768, action_size: int = 8, learning_rate: float = 0.001, base_dir: Optional[Path] = None):
        """
        Initialize the emotional learning system.
        
        Args:
            state_size: Size of the emotional state vector
            action_size: Number of possible emotional responses
            learning_rate: Learning rate for the neural network
            base_dir: Base directory for saving/loading data (defaults to parent directory of this file)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.emotional_patterns = []
        self.pattern_weights = {}
        self.learning_history = []
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir if base_dir is not None else Path(__file__).resolve().parent
        self.pattern_threshold = 0.3
        self.pattern_memory = []
        self.discount_factor = 0.95
        self.q_table = {}
        self.experience_buffer = []
        self.buffer_size = 10000
        self.min_experiences = 32
        self.strategies = {
            'defensive': {'response_modifier': 0.5},
            'aggressive': {'response_modifier': 1.5},
            'balanced': {'response_modifier': 1.0}
        }
        self.current_strategy = 'balanced'
        
    def initialize(self):
        """No-op initializer for compatibility with main system."""
        self.logger.info("EmotionalLearningSystem initialized (no-op)")
        return True
        
    def cleanup(self):
        """No-op cleanup for compatibility with main system."""
        self.logger.info("EmotionalLearningSystem cleanup (no-op)")
        return True
        
    @property
    def patterns(self):
        """Property to access emotional patterns for compatibility."""
        return self.emotional_patterns
        
    def _build_model(self) -> nn.Module:
        """
        Build the neural network model for emotional learning.
        
        Returns:
            Neural network model
        """
        model = nn.Sequential(
            nn.Linear(self.state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.action_size)
        )
        return model
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in memory.
        
        Args:
            state: Current emotional state
            action: Action taken
            reward: Reward received
            next_state: Next emotional state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray) -> int:
        """
        Choose an action based on the current state.
        
        Args:
            state: Current emotional state
            
        Returns:
            Selected action index
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state_tensor)
            return torch.argmax(act_values).item()
            
    def replay(self, batch_size: int) -> None:
        """
        Train the model using experience replay.
        
        Args:
            batch_size: Number of experiences to sample
        """
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([x[0] for x in minibatch]))
        actions = torch.LongTensor(np.array([x[1] for x in minibatch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch]))
        next_states = torch.FloatTensor(np.array([x[3] for x in minibatch]))
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch]))
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> float:
        """
        Learn from an emotional interaction.
        
        Args:
            interaction_data: Dictionary containing interaction information including:
                - emotional_state: Current emotional state
                - response_index: Index of the response taken
                - next_emotional_state: Resulting emotional state
                - response_appropriateness: How appropriate the response was
                - emotional_stability: Stability of the emotional state
                - empathy_effectiveness: How effective the empathy was
                - emotional_intensity: Intensity of the emotional response
                - personality_consistency: How consistent with personality
            
        Returns:
            Learning reward value
        """
        try:
            # Extract emotional state and response
            current_state = np.array(interaction_data['emotional_state'])
            action = interaction_data['response_index']
            next_state = np.array(interaction_data['next_emotional_state'])
            
            # Calculate reward based on interaction outcome
            reward = self._calculate_reward(interaction_data)
            
            # Store experience
            self.remember(current_state, action, reward, next_state, False)
            
            # Train on batch of experiences
            self.replay(32)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error in emotional learning: {str(e)}")
            return 0.0
            
    def _calculate_reward(self, interaction_data: Dict[str, Any]) -> float:
        """
        Calculate reward for an interaction based on various factors.
        
        Args:
            interaction_data: Dictionary containing interaction information
            
        Returns:
            Calculated reward value
        """
        try:
            reward = 0.0
            
            # Reward for appropriate emotional response
            if interaction_data.get('response_appropriateness', 0) > 0.7:
                reward += 1.0
                
            # Reward for emotional stability
            if interaction_data.get('emotional_stability', 0) > 0.6:
                reward += 0.5
                
            # Reward for successful empathy
            if interaction_data.get('empathy_effectiveness', 0) > 0.8:
                reward += 1.0
                
            # Penalty for emotional extremes
            if interaction_data.get('emotional_intensity', 0) > 0.9:
                reward -= 0.5
                
            # Reward for personality consistency
            if interaction_data.get('personality_consistency', 0) > 0.7:
                reward += 0.5
                
            return max(-1.0, min(1.0, reward))  # Normalize reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0
            
    def get_emotional_strategy(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get the current emotional strategy for a given state.
        
        Args:
            state: Current emotional state
            
        Returns:
            Dictionary of emotional response probabilities
        """
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                probabilities = torch.softmax(q_values, dim=1).squeeze().numpy()
                
                return {
                    'joy': float(probabilities[0]),
                    'sadness': float(probabilities[1]),
                    'anger': float(probabilities[2]),
                    'fear': float(probabilities[3]),
                    'surprise': float(probabilities[4]),
                    'disgust': float(probabilities[5]),
                    'neutral': float(probabilities[6]),
                    'complex': float(probabilities[7])
                }
                
        except Exception as e:
            self.logger.error(f"Error getting emotional strategy: {str(e)}")
            return {
                'joy': 0.125,
                'sadness': 0.125,
                'anger': 0.125,
                'fear': 0.125,
                'surprise': 0.125,
                'disgust': 0.125,
                'neutral': 0.125,
                'complex': 0.125
            }
            
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        try:
            torch.save(self.model.state_dict(), path)
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    def load_model(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        try:
            checkpoint = torch.load(path, weights_only=True)
            self.model.load_state_dict(checkpoint)
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            
    def detect_patterns(self, emotional_history: List[Dict[str, float]]) -> List[Dict]:
        """
        Detect patterns in emotional history.
        
        Args:
            emotional_history: List of emotional states over time
            
        Returns:
            List of detected patterns
        """
        try:
            patterns = []
            if len(emotional_history) < 3:
                return patterns
                
            # Calculate differences between consecutive states
            differences = []
            for i in range(1, len(emotional_history)):
                diff = {
                    k: emotional_history[i][k] - emotional_history[i-1][k]
                    for k in emotional_history[i].keys()
                    if k != 'timestamp'
                }
                differences.append(diff)
                
            # Look for repeating patterns
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
            
    def _is_pattern_significant(self, pattern: List[Dict[str, float]]) -> bool:
        """
        Check if a pattern is significant enough to be considered.
        
        Args:
            pattern: List of emotional state differences
            
        Returns:
            Whether the pattern is significant
        """
        try:
            # Calculate average magnitude of changes
            magnitudes = []
            for diff in pattern:
                magnitude = np.sqrt(sum(v * v for v in diff.values()))
                magnitudes.append(magnitude)
                
            avg_magnitude = np.mean(magnitudes)
            return avg_magnitude > 0.3  # Threshold for significance
            
        except Exception as e:
            self.logger.error(f"Error checking pattern significance: {str(e)}")
            return False
            
    def _calculate_pattern_confidence(self, pattern: List[Dict[str, float]]) -> float:
        """
        Calculate confidence score for a detected pattern.
        
        Args:
            pattern: List of emotional state differences
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Compare with existing patterns
            if not self.emotional_patterns:
                return 1.0
                
            similarities = []
            for stored_pattern in self.emotional_patterns:
                similarity = self._calculate_pattern_similarity(pattern, stored_pattern['pattern'])
                similarities.append(similarity)
                
            return max(similarities) if similarities else 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {str(e)}")
            return 0.0
            
    def _calculate_pattern_similarity(self, pattern1: List[Dict[str, float]], 
                                   pattern2: List[Dict[str, float]]) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if len(pattern1) != len(pattern2):
                return 0.0
                
            similarities = []
            for p1, p2 in zip(pattern1, pattern2):
                # Calculate cosine similarity between state differences
                v1 = np.array(list(p1.values()))
                v2 = np.array(list(p2.values()))
                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                similarities.append(similarity)
                
            return np.mean(similarities)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {str(e)}")
            return 0.0
            
    def adapt_strategy(self, emotional_state: Dict[str, float], 
                      interaction_history: List[Dict]) -> str:
        """
        Adapt emotional response strategy based on current state and history.
        
        Args:
            emotional_state: Current emotional state
            interaction_history: History of interactions
            
        Returns:
            Selected strategy ('defensive', 'aggressive', or 'balanced')
        """
        try:
            # Calculate emotional stability
            stability = self._calculate_emotional_stability(emotional_state)
            
            # Calculate interaction success rate
            success_rate = self._calculate_success_rate(interaction_history)
            
            # Choose strategy based on stability and success
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
        """
        Calculate emotional stability score.
        
        Args:
            emotional_state: Current emotional state
            
        Returns:
            Stability score between 0 and 1
        """
        try:
            # Calculate variance of emotional values
            values = [v for k, v in emotional_state.items() if k != 'timestamp']
            variance = np.var(values)
            
            # Convert variance to stability score (0-1)
            # Higher variance = lower stability
            stability = 1.0 / (1.0 + np.exp(variance * 10 - 2))
            return stability
            
        except Exception as e:
            self.logger.error(f"Error calculating emotional stability: {str(e)}")
            return 0.5
            
    def _calculate_success_rate(self, interaction_history: List[Dict]) -> float:
        """
        Calculate success rate from interaction history.
        
        Args:
            interaction_history: History of interactions
            
        Returns:
            Success rate between 0 and 1
        """
        try:
            if not interaction_history:
                return 0.5
                
            # Count successful interactions (positive sentiment)
            successful = sum(1 for interaction in interaction_history 
                           if interaction.get('sentiment', 0) > 0)
            
            return successful / len(interaction_history)
            
        except Exception as e:
            self.logger.error(f"Error calculating success rate: {str(e)}")
            return 0.5
            
    def adapt_emotional_response(self, emotional_state: Dict[str, float],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt emotional response using learned patterns.
        
        Args:
            emotional_state: Current emotional state
            context: Interaction context
            
        Returns:
            Adapted emotional response
        """
        try:
            # Convert state to vector
            state_vector = np.array([
                emotional_state.get('joy', 0.0),
                emotional_state.get('sadness', 0.0),
                emotional_state.get('anger', 0.0),
                emotional_state.get('fear', 0.0),
                emotional_state.get('surprise', 0.0),
                emotional_state.get('disgust', 0.0),
                emotional_state.get('neutral', 0.0),
                emotional_state.get('complex', 0.0)
            ])
            
            # Get strategy probabilities
            strategy = self.get_emotional_strategy(state_vector)
            
            # Generate response based on strategy
            response = {
                'text': self._generate_response_text(strategy, context),
                'style': {
                    'expressiveness': max(strategy.values()),
                    'empathy': strategy.get('complex', 0.0)
                }
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error adapting emotional response: {str(e)}")
            return {
                'text': "I understand.",
                'style': {'expressiveness': 0.5, 'empathy': 0.5}
            }
            
    def _generate_response_text(self, strategy: Dict[str, float],
                              context: Dict[str, Any]) -> str:
        """
        Generate response text based on emotional strategy.
        
        Args:
            strategy: Emotional strategy probabilities
            context: Interaction context
            
        Returns:
            Generated response text
        """
        try:
            # Get dominant emotion
            dominant_emotion = max(strategy.items(), key=lambda x: x[1])[0]
            
            # Response templates
            templates = {
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
            
            # Select template based on context and emotion
            if context.get('interaction_type') == 'positive':
                return random.choice(templates['joy'])
            elif context.get('interaction_type') == 'negative':
                return random.choice(templates['sadness'])
            else:
                return random.choice(templates[dominant_emotion])
                
        except Exception as e:
            self.logger.error(f"Error generating response text: {str(e)}")
            return "I understand."
            
    def save_learning_state(self, path: str) -> None:
        """
        Save the learning state to a file.
        
        Args:
            path: Path to save the learning state
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'emotional_patterns': self.emotional_patterns,
                'pattern_weights': self.pattern_weights,
                'learning_history': self.learning_history,
                'pattern_memory': self.pattern_memory,
                'q_table': self.q_table,
                'current_strategy': self.current_strategy
            }, path)
        except Exception as e:
            self.logger.error(f"Error saving learning state: {str(e)}")
            
    def load_learning_state(self, path: str) -> None:
        """
        Load the learning state from a file.
        
        Args:
            path: Path to load the learning state from
        """
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.emotional_patterns = checkpoint['emotional_patterns']
            self.pattern_weights = checkpoint['pattern_weights']
            self.learning_history = checkpoint['learning_history']
            self.pattern_memory = checkpoint['pattern_memory']
            self.q_table = checkpoint['q_table']
            self.current_strategy = checkpoint['current_strategy']
        except Exception as e:
            self.logger.error(f"Error loading learning state: {str(e)}")
            
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status.
        
        Returns:
            Dict[str, Any]: Learning status information
        """
        try:
            return {
                'patterns_count': len(self.emotional_patterns),
                'history_size': len(self.learning_history),
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'pattern_weights': self.pattern_weights,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting learning status: {e}")
            return {
                'error': str(e)
            }

    def get_current_state(self):
        """
        Returns the current state of the learning system.
        
        Returns:
            dict: A dictionary containing the current learning state metrics
        """
        try:
            return {
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'model_state': {
                    'input_size': self.state_size,
                    'output_size': self.action_size,
                    'training_steps': self.training_steps
                },
                'last_loss': self.last_loss if hasattr(self, 'last_loss') else None,
                'last_reward': self.last_reward if hasattr(self, 'last_reward') else None
            }
        except Exception as e:
            logging.error(f"Error getting learning state: {str(e)}")
            return {
                'learning_rate': 0.0,
                'epsilon': 0.0,
                'memory_size': 0,
                'model_state': None,
                'last_loss': None,
                'last_reward': None
            }

    def reset_learning(self) -> Dict[str, Any]:
        """Reset learning state.
        
        Returns:
            Dict[str, Any]: Reset status
        """
        try:
            self.emotional_patterns = []
            self.pattern_weights = {}
            self.learning_history = []
            self.q_table = {}
            self.experience_buffer = []
            self._save_patterns()
            
            return {
                'success': True,
                'message': 'Learning state reset successfully'
            }
        except Exception as e:
            self.logger.error(f"Error resetting learning state: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _load_learning_data(self):
        """Load existing learning data from disk."""
        try:
            data_file = self.base_dir / "learning_data.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.pattern_memory = data.get('patterns', [])
                    self.current_strategy = data.get('strategy', 'balanced')
        except Exception as e:
            self.logger.error(f"Error loading learning data: {str(e)}")
    
    def _save_learning_data(self) -> None:
        """
        Save the current learning state to disk.
        This includes model weights, patterns, and other learning parameters.
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data/learning', exist_ok=True)
            
            # Save model weights
            torch.save(self.model.state_dict(), 'data/learning/model_weights.pt')
            
            # Save learning parameters
            learning_data = {
                'epsilon': self.epsilon,
                'emotional_patterns': self.emotional_patterns,
                'pattern_weights': self.pattern_weights,
                'learning_history': self.learning_history,
                'pattern_memory': self.pattern_memory,
                'q_table': self.q_table,
                'current_strategy': self.current_strategy
            }
            
            with open('data/learning/learning_state.json', 'w') as f:
                json.dump(learning_data, f)
                
        except Exception as e:
            self.logger.error(f"Error saving learning data: {str(e)}")
            raise
    
    def _process_batch(self, batch: List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]) -> torch.Tensor:
        """Process a batch of experiences for training.
        
        Args:
            batch: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            torch.Tensor: Loss value
        """
        if not batch:
            raise ValueError("Empty batch provided")
            
        try:
            # Unpack batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.cat(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.cat(next_states)
            dones = torch.FloatTensor(dones)
            
            # Get current Q values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            
            # Get next Q values from target network
            next_q_values = self.model(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise

    def _serialize_learning_data(self, data: Dict[str, Any]) -> str:
        """Serialize learning data to JSON string.
        
        Args:
            data: Dictionary containing learning data
            
        Returns:
            str: JSON serialized data
        """
        try:
            # Convert tensors to lists
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    serializable_data[key] = value.tolist()
                elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                    serializable_data[key] = [v.tolist() for v in value]
                else:
                    serializable_data[key] = value
                    
            return json.dumps(serializable_data)
            
        except Exception as e:
            self.logger.error(f"Error serializing learning data: {e}")
            raise

    def _deserialize_learning_data(self, data_str: str) -> Dict[str, Any]:
        """Deserialize learning data from JSON string.
        
        Args:
            data_str: JSON serialized data string
            
        Returns:
            Dict[str, Any]: Deserialized data
        """
        try:
            data = json.loads(data_str)
            
            # Convert lists back to tensors where appropriate
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], list):  # Nested list (tensor)
                        data[key] = torch.tensor(value)
                    elif key in ['state', 'next_state']:  # State tensors
                        data[key] = torch.tensor(value)
                        
            return data
            
        except Exception as e:
            self.logger.error(f"Error deserializing learning data: {e}")
            raise

    def learn_from_experience(self, 
                            emotion: str,
                            intensity: float,
                            context: Dict[str, Any],
                            outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from an emotional experience.
        
        Args:
            emotion: Type of emotion experienced
            intensity: Intensity of the emotion
            context: Context of the experience
            outcome: Outcome of the experience
            
        Returns:
            Dict[str, Any]: Learning results
        """
        try:
            # Create experience record
            experience = {
                'emotion': emotion,
                'intensity': intensity,
                'context': context,
                'outcome': outcome,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to learning history
            self.learning_history.append(experience)
            
            # Update pattern weights
            self._update_pattern_weights(experience)
            
            # Detect new patterns
            new_patterns = self._detect_patterns(experience)
            if new_patterns:
                self.emotional_patterns.extend(new_patterns)
                self._save_patterns()
                
            return {
                'success': True,
                'patterns_updated': len(new_patterns),
                'weights_updated': True
            }
            
        except Exception as e:
            self.logger.error(f"Error learning from experience: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _update_pattern_weights(self, experience: Dict[str, Any]) -> None:
        """Update weights of emotional patterns based on experience.
        
        Args:
            experience: Experience dictionary
        """
        try:
            # Calculate reward based on outcome
            reward = self._calculate_reward(experience['outcome'])
            
            # Update weights for matching patterns
            for pattern in self.emotional_patterns:
                if self._pattern_matches_experience(pattern, experience):
                    current_weight = self.pattern_weights.get(pattern['id'], 0.5)
                    new_weight = current_weight + self.learning_rate * (reward - current_weight)
                    self.pattern_weights[pattern['id']] = max(0.0, min(1.0, new_weight))
                    
        except Exception as e:
            self.logger.error(f"Error updating pattern weights: {e}")
            
    def _calculate_reward(self, outcome: Dict[str, Any]) -> float:
        """Calculate reward from outcome.
        
        Args:
            outcome: Outcome dictionary
            
        Returns:
            float: Reward value between 0 and 1
        """
        try:
            # Extract relevant metrics
            success = outcome.get('success', False)
            satisfaction = outcome.get('satisfaction', 0.5)
            adaptation = outcome.get('adaptation', 0.5)
            
            # Calculate weighted reward
            reward = (
                0.4 * float(success) +
                0.3 * satisfaction +
                0.3 * adaptation
            )
            
            return max(0.0, min(1.0, reward))
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.5
            
    def _pattern_matches_experience(self, pattern: Dict[str, Any], experience: Dict[str, Any]) -> bool:
        """Check if pattern matches experience.
        
        Args:
            pattern: Pattern dictionary
            experience: Experience dictionary
            
        Returns:
            bool: True if pattern matches experience
        """
        try:
            # Check emotion match
            if pattern['emotion'] != experience['emotion']:
                return False
                
            # Check intensity range
            if not (pattern['intensity_min'] <= experience['intensity'] <= pattern['intensity_max']):
                return False
                
            # Check context similarity
            context_similarity = self._calculate_context_similarity(
                pattern['context'],
                experience['context']
            )
            if context_similarity < 0.7:  # Threshold for context matching
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking pattern match: {e}")
            return False
            
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts.
        
        Args:
            context1: First context dictionary
            context2: Second context dictionary
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Get common keys
            common_keys = set(context1.keys()) & set(context2.keys())
            if not common_keys:
                return 0.0
                
            # Calculate similarity for each key
            similarities = []
            for key in common_keys:
                if isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
                    # Numeric similarity
                    diff = abs(context1[key] - context2[key])
                    similarities.append(1.0 - min(1.0, diff))
                else:
                    # String or other type similarity
                    similarities.append(1.0 if context1[key] == context2[key] else 0.0)
                    
            return sum(similarities) / len(similarities)
            
        except Exception as e:
            self.logger.error(f"Error calculating context similarity: {e}")
            return 0.0
            
    def _detect_patterns(self, experience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect new patterns from experience.
        
        Args:
            experience: Experience dictionary
            
        Returns:
            List[Dict[str, Any]]: List of new patterns
        """
        try:
            new_patterns = []
            
            # Check for similar experiences
            similar_experiences = [
                exp for exp in self.learning_history[-10:]  # Look at recent history
                if exp['emotion'] == experience['emotion']
                and abs(exp['intensity'] - experience['intensity']) < 0.2
            ]
            
            if len(similar_experiences) >= 3:  # Minimum experiences for pattern
                # Create pattern from similar experiences
                pattern = {
                    'id': f"pattern_{len(self.emotional_patterns)}",
                    'emotion': experience['emotion'],
                    'intensity_min': min(exp['intensity'] for exp in similar_experiences),
                    'intensity_max': max(exp['intensity'] for exp in similar_experiences),
                    'context': self._extract_common_context(similar_experiences),
                    'outcomes': [exp['outcome'] for exp in similar_experiences],
                    'created_at': datetime.now().isoformat()
                }
                new_patterns.append(pattern)
                
            return new_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
            
    def _extract_common_context(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common context from similar experiences.
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            Dict[str, Any]: Common context dictionary
        """
        try:
            if not experiences:
                return {}
                
            # Get all context keys
            all_keys = set()
            for exp in experiences:
                all_keys.update(exp['context'].keys())
                
            # Find common values
            common_context = {}
            for key in all_keys:
                values = [exp['context'].get(key) for exp in experiences if key in exp['context']]
                if all(v == values[0] for v in values):
                    common_context[key] = values[0]
                    
            return common_context
            
        except Exception as e:
            self.logger.error(f"Error extracting common context: {e}")
            return {}
            
    def _save_patterns(self) -> None:
        """
        Save emotional patterns to disk.
        """
        try:
            # Create data directory if it doesn't exist
            data_dir = self.base_dir / 'learning'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save patterns
            patterns_data = {
                'emotional_patterns': self.emotional_patterns,
                'pattern_weights': self.pattern_weights,
                'pattern_memory': self.pattern_memory,
                'pattern_threshold': self.pattern_threshold
            }
            
            patterns_path = data_dir / 'patterns.json'
            with open(patterns_path, 'w') as f:
                json.dump(patterns_data, f)
            
        except Exception as e:
            self.logger.error(f"Error saving patterns: {str(e)}")
            raise             # Save patterns
            patterns_data = {
                'emotional_patterns': self.emotional_patterns,
                'pattern_weights': self.pattern_weights,
                'pattern_memory': self.pattern_memory,
                'pattern_threshold': self.pattern_threshold
            }
            
            patterns_path = data_dir / 'patterns.json'
            with open(patterns_path, 'w') as f:
                json.dump(patterns_data, f)
            
        except Exception as e:
            self.logger.error(f"Error saving patterns: {str(e)}")
            raise 
