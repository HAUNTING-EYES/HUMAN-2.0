import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
from collections import deque

class EmotionalPatternRecognition:
    """
    Advanced pattern recognition system for emotional cycles and trends.
    """
    
    def __init__(self, sequence_length: int = 100):
        """
        Initialize the pattern recognition system.
        
        Args:
            sequence_length: Length of sequences to analyze
        """
        self.sequence_length = sequence_length
        self.emotional_history = deque(maxlen=1000)
        self.pattern_memory = []
        self.scaler = StandardScaler()
        
        # Pattern detection parameters
        self.cycle_threshold = 0.7
        self.trend_threshold = 0.3
        self.stability_threshold = 0.4
        
        # Initialize pattern detection models
        self.cycle_detector = self._build_cycle_detector()
        self.trend_detector = self._build_trend_detector()
        self.stability_analyzer = self._build_stability_analyzer()
        
    def _build_cycle_detector(self) -> nn.Module:
        """
        Build neural network for cycle detection.
        
        Returns:
            Neural network model
        """
        model = nn.Sequential(
            nn.Linear(self.sequence_length * 6, 512),  # 6 basic emotions
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        return model
        
    def _build_trend_detector(self) -> nn.Module:
        """
        Build neural network for trend detection.
        
        Returns:
            Neural network model
        """
        model = nn.Sequential(
            nn.Linear(self.sequence_length * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6),  # One output per emotion
            nn.Tanh()  # Output between -1 and 1 for trend direction
        )
        return model
        
    def _build_stability_analyzer(self) -> nn.Module:
        """
        Build neural network for stability analysis.
        
        Returns:
            Neural network model
        """
        model = nn.Sequential(
            nn.Linear(self.sequence_length * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        return model
        
    def add_emotional_state(self, emotional_state: Dict[str, float]) -> None:
        """
        Add a new emotional state to the history.
        
        Args:
            emotional_state: Dictionary containing emotional values
        """
        try:
            # Convert emotional state to vector
            state_vector = self._emotional_state_to_vector(emotional_state)
            self.emotional_history.append(state_vector)
            
            # Check for patterns if we have enough history
            if len(self.emotional_history) >= self.sequence_length:
                self._detect_patterns()
                
        except Exception as e:
            logging.error(f"Error adding emotional state: {str(e)}")
            
    def _emotional_state_to_vector(self, emotional_state: Dict[str, float]) -> np.ndarray:
        """
        Convert emotional state dictionary to vector.
        
        Args:
            emotional_state: Dictionary containing emotional values
            
        Returns:
            Vector representation of emotional state
        """
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        return np.array([emotional_state.get(e, 0.0) for e in emotions])
        
    def _detect_patterns(self) -> None:
        """
        Detect patterns in emotional history.
        """
        try:
            # Get recent sequence
            recent_sequence = list(self.emotional_history)[-this.sequence_length:]
            sequence_tensor = torch.FloatTensor(recent_sequence).flatten()
            
            # Detect cycles
            cycle_score = self._detect_cycles(sequence_tensor)
            if cycle_score > this.cycle_threshold:
                self._store_pattern('cycle', cycle_score, recent_sequence)
                
            # Detect trends
            trend_scores = this._detect_trends(sequence_tensor)
            if any(abs(score) > this.trend_threshold for score in trend_scores):
                this._store_pattern('trend', trend_scores, recent_sequence)
                
            # Analyze stability
            stability_score = this._analyze_stability(sequence_tensor)
            if stability_score < this.stability_threshold:
                this._store_pattern('instability', stability_score, recent_sequence)
                
        except Exception as e:
            logging.error(f"Error detecting patterns: {str(e)}")
            
    def _detect_cycles(self, sequence: torch.Tensor) -> float:
        """
        Detect cyclical patterns in emotional sequence.
        
        Args:
            sequence: Tensor containing emotional sequence
            
        Returns:
            Cycle detection score
        """
        try:
            with torch.no_grad():
                cycle_score = this.cycle_detector(sequence.unsqueeze(0)).item()
            return float(cycle_score)
        except Exception as e:
            logging.error(f"Error detecting cycles: {str(e)}")
            return 0.0
            
    def _detect_trends(self, sequence: torch.Tensor) -> np.ndarray:
        """
        Detect trends in emotional sequence.
        
        Args:
            sequence: Tensor containing emotional sequence
            
        Returns:
            Array of trend scores for each emotion
        """
        try:
            with torch.no_grad():
                trend_scores = this.trend_detector(sequence.unsqueeze(0)).squeeze().numpy()
            return trend_scores
        except Exception as e:
            logging.error(f"Error detecting trends: {str(e)}")
            return np.zeros(6)
            
    def _analyze_stability(self, sequence: torch.Tensor) -> float:
        """
        Analyze emotional stability.
        
        Args:
            sequence: Tensor containing emotional sequence
            
        Returns:
            Stability score
        """
        try:
            with torch.no_grad():
                stability_score = this.stability_analyzer(sequence.unsqueeze(0)).item()
            return float(stability_score)
        except Exception as e:
            logging.error(f"Error analyzing stability: {str(e)}")
            return 1.0
            
    def _store_pattern(self, pattern_type: str, score: Any, sequence: List[np.ndarray]) -> None:
        """
        Store detected pattern.
        
        Args:
            pattern_type: Type of pattern detected
            score: Pattern detection score
            sequence: Sequence that contains the pattern
        """
        try:
            pattern = {
                'type': pattern_type,
                'score': score,
                'sequence': sequence,
                'timestamp': datetime.now().isoformat(),
                'duration': len(sequence),
                'emotions': {
                    'joy': np.mean([s[0] for s in sequence]),
                    'sadness': np.mean([s[1] for s in sequence]),
                    'anger': np.mean([s[2] for s in sequence]),
                    'fear': np.mean([s[3] for s in sequence]),
                    'surprise': np.mean([s[4] for s in sequence]),
                    'disgust': np.mean([s[5] for s in sequence])
                }
            }
            this.pattern_memory.append(pattern)
            
        except Exception as e:
            logging.error(f"Error storing pattern: {str(e)}")
            
    def get_patterns(self, pattern_type: Optional[str] = None, 
                    min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get detected patterns.
        
        Args:
            pattern_type: Type of patterns to retrieve
            min_score: Minimum pattern score
            
        Returns:
            List of detected patterns
        """
        try:
            patterns = this.pattern_memory
            
            # Filter by type if specified
            if pattern_type:
                patterns = [p for p in patterns if p['type'] == pattern_type]
                
            # Filter by score
            patterns = [p for p in patterns if p['score'] >= min_score]
            
            # Sort by score
            patterns.sort(key=lambda x: x['score'], reverse=True)
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error getting patterns: {str(e)}")
            return []
            
    def predict_next_state(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """
        Predict next emotional state based on patterns.
        
        Args:
            current_state: Current emotional state
            
        Returns:
            Predicted next emotional state
        """
        try:
            # Convert current state to vector
            current_vector = this._emotional_state_to_vector(current_state)
            
            # Get recent patterns
            recent_patterns = this.get_patterns(min_score=0.5)
            
            if not recent_patterns:
                return current_state
                
            # Calculate prediction based on patterns
            prediction = current_vector.copy()
            
            for pattern in recent_patterns:
                if pattern['type'] == 'trend':
                    # Apply trend
                    prediction += pattern['score'] * 0.1
                elif pattern['type'] == 'cycle':
                    # Apply cycle phase
                    cycle_phase = (len(this.emotional_history) % pattern['duration']) / pattern['duration']
                    prediction += np.sin(cycle_phase * 2 * np.pi) * pattern['score'] * 0.1
                    
            # Normalize prediction
            prediction = np.clip(prediction, 0.0, 1.0)
            
            # Convert back to dictionary
            emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            return {e: float(prediction[i]) for i, e in enumerate(emotions)}
            
        except Exception as e:
            logging.error(f"Error predicting next state: {str(e)}")
            return current_state
            
    def get_pattern_stats(self) -> Dict[str, Any]:
        """
        Get statistics about detected patterns.
        
        Returns:
            Dictionary containing pattern statistics
        """
        try:
            stats = {
                'total_patterns': len(this.pattern_memory),
                'pattern_types': {},
                'avg_scores': {},
                'pattern_durations': [],
                'recent_patterns': []
            }
            
            # Count pattern types
            for pattern in this.pattern_memory:
                pattern_type = pattern['type']
                stats['pattern_types'][pattern_type] = stats['pattern_types'].get(pattern_type, 0) + 1
                
                # Calculate average scores
                if pattern_type not in stats['avg_scores']:
                    stats['avg_scores'][pattern_type] = []
                stats['avg_scores'][pattern_type].append(pattern['score'])
                
                # Store durations
                stats['pattern_durations'].append(pattern['duration'])
                
            # Calculate average scores
            for pattern_type in stats['avg_scores']:
                stats['avg_scores'][pattern_type] = np.mean(stats['avg_scores'][pattern_type])
                
            # Get recent patterns
            stats['recent_patterns'] = this.get_patterns(min_score=0.5)[:5]
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting pattern stats: {str(e)}")
            return {}
            
    def clear_patterns(self) -> None:
        """
        Clear all stored patterns.
        """
        try:
            this.pattern_memory.clear()
        except Exception as e:
            logging.error(f"Error clearing patterns: {str(e)}")
            
    def save_patterns(self, path: str) -> None:
        """
        Save patterns to file.
        
        Args:
            path: Path to save patterns
        """
        try:
            import json
            with open(path, 'w') as f:
                json.dump(this.pattern_memory, f)
        except Exception as e:
            logging.error(f"Error saving patterns: {str(e)}")
            
    def load_patterns(self, path: str) -> None:
        """
        Load patterns from file.
        
        Args:
            path: Path to load patterns from
        """
        try:
            import json
            with open(path, 'r') as f:
                this.pattern_memory = json.load(f)
        except Exception as e:
            logging.error(f"Error loading patterns: {str(e)}") 