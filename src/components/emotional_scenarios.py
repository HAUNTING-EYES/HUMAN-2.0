import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from collections import deque
import logging
from datetime import datetime
import json
from pathlib import Path
import tempfile

class ConflictResolver(nn.Module):
    """Neural network for resolving emotional conflicts."""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 32),  # 6 emotions * 2 (current + target)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)    # 6 emotions output
        )
        
    def forward(self, current_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        combined = torch.cat([current_state, target_state], dim=0)
        return torch.sigmoid(self.network(combined))

class TransitionManager(nn.Module):
    """Neural network for managing emotional transitions."""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 32),  # 6 emotions * 2 (current + target)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)    # 6 emotions output
        )
        
    def forward(self, current_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        combined = torch.cat([current_state, target_state], dim=0)
        return torch.sigmoid(self.network(combined))

class ScenarioClassifier(nn.Module):
    """Neural network for classifying emotional scenarios."""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 32),   # 6 emotions input
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)   # 10 scenario types output
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return torch.softmax(self.network(state), dim=0)

class ComplexEmotionalScenarios:
    """Handler for complex emotional scenarios."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the scenarios handler."""
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.mkdtemp())
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize neural networks
        self.conflict_resolver = ConflictResolver()
        self.transition_manager = TransitionManager()
        self.scenario_classifier = ScenarioClassifier()
        
        # Initialize scenario tracking
        self.scenario_history = []
        self.active_scenarios = []
        
        # Define scenario types
        self.scenario_types = [
            'simple', 'conflict', 'transition', 'complex',
            'stable', 'unstable', 'intense', 'mild',
            'positive', 'negative'
        ]
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load existing scenarios
        self._load_scenarios()
        
    def process_scenario(self, current_state: Dict[str, float], 
                        context: Dict[str, Any]) -> Dict[str, float]:
        """Process an emotional scenario."""
        try:
            # 1. Classify scenario
            scenario_type = self._classify_scenario(current_state)
            
            # 2. Detect conflicts
            conflicts = self._detect_conflicts(current_state)
            
            # 3. Resolve conflicts if any
            if conflicts:
                current_state = self._resolve_conflicts(current_state, conflicts)
                
            # 4. Manage transitions
            current_state = self._manage_transitions(current_state, context)
            
            # 5. Store scenario
            self._store_scenario({
                'type': scenario_type,
                'emotional_state': current_state.copy(),
                'conflicts': conflicts,
                'context': context,
                'timestamp': datetime.now().isoformat()
            })
            
            return current_state
            
        except Exception as e:
            self.logger.error(f"Error processing scenario: {str(e)}")
            return current_state
            
    def _classify_scenario(self, state: Dict[str, float]) -> str:
        """Classify the emotional scenario."""
        try:
            # Convert state to tensor
            state_tensor = torch.FloatTensor([
                state.get(emotion, 0.0)
                for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            ])
            
            # Get classification probabilities
            with torch.no_grad():
                probs = self.scenario_classifier(state_tensor)
                
            # Return most likely scenario type
            return self.scenario_types[torch.argmax(probs).item()]
            
        except Exception as e:
            self.logger.error(f"Error classifying scenario: {str(e)}")
            return 'simple'
            
    def _detect_conflicts(self, state: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect emotional conflicts."""
        conflicts = []
        
        try:
            # Check for opposing emotions
            opposing_pairs = [
                ('joy', 'sadness'),
                ('anger', 'fear'),
                ('surprise', 'disgust')
            ]
            
            for emotion1, emotion2 in opposing_pairs:
                if state.get(emotion1, 0.0) > 0.6 and state.get(emotion2, 0.0) > 0.6:
                    conflicts.append({
                        'type': 'opposition',
                        'emotions': [emotion1, emotion2],
                        'strength': min(state[emotion1], state[emotion2])
                    })
                    
            # Check for emotional overload
            active_emotions = sum(1 for v in state.values() if v > 0.6)
            if active_emotions > 3:
                conflicts.append({
                    'type': 'overload',
                    'emotions': [e for e, v in state.items() if v > 0.6],
                    'strength': active_emotions / len(state)
                })
                
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Error detecting conflicts: {str(e)}")
            return []
            
    def _resolve_conflicts(self, state: Dict[str, float], 
                         conflicts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Resolve emotional conflicts."""
        try:
            if not conflicts:
                return state
                
            # Convert state to tensor
            current_tensor = torch.FloatTensor([
                state.get(emotion, 0.0)
                for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            ])
            
            # Create target state (balanced)
            target_tensor = torch.FloatTensor([0.5] * 6)
            
            # Resolve conflicts using neural network
            with torch.no_grad():
                resolved_tensor = self.conflict_resolver(current_tensor, target_tensor)
                
            # Convert back to dictionary
            return {
                emotion: float(value)
                for emotion, value in zip(
                    ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                    resolved_tensor
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error resolving conflicts: {str(e)}")
            return state
            
    def _manage_transitions(self, state: Dict[str, float], 
                          context: Dict[str, Any]) -> Dict[str, float]:
        """Manage emotional transitions."""
        try:
            target_state = context.get('target_emotional_state', state)
            
            # Convert states to tensors
            current_tensor = torch.FloatTensor([
                state.get(emotion, 0.0)
                for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            ])
            
            target_tensor = torch.FloatTensor([
                target_state.get(emotion, 0.0)
                for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            ])
            
            # Manage transition using neural network
            with torch.no_grad():
                transitioned_tensor = self.transition_manager(current_tensor, target_tensor)
                
            # Convert back to dictionary
            return {
                emotion: float(value)
                for emotion, value in zip(
                    ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                    transitioned_tensor
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error managing transitions: {str(e)}")
            return state
            
    def _store_scenario(self, scenario: Dict[str, Any]) -> None:
        """Store a processed scenario."""
        try:
            self.scenario_history.append(scenario)
            
            # Update active scenarios
            self.active_scenarios = self.scenario_history[-10:]
            
            # Save to file
            self._save_scenarios()
            
        except Exception as e:
            self.logger.error(f"Error storing scenario: {str(e)}")
            
    def get_scenario_stats(self) -> Dict[str, Any]:
        """Get scenario statistics."""
        try:
            if not self.scenario_history:
                return {
                    'total_scenarios': 0,
                    'scenario_types': {},
                    'conflict_stats': {
                        'total_conflicts': 0,
                        'avg_conflicts_per_scenario': 0.0
                    },
                    'transition_stats': {
                        'total_transitions': 0,
                        'avg_transition_time': 0.0
                    }
                }
                
            # Calculate statistics
            total_scenarios = len(self.scenario_history)
            scenario_types = {}
            total_conflicts = 0
            total_transitions = 0
            
            for scenario in self.scenario_history:
                # Count scenario types
                scenario_type = scenario['type']
                scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
                
                # Count conflicts
                if scenario['conflicts']:
                    total_conflicts += len(scenario['conflicts'])
                    
                # Count transitions
                if 'context' in scenario and 'target_emotional_state' in scenario['context']:
                    target_state = scenario['context']['target_emotional_state']
                    current_state = scenario['emotional_state']
                    if target_state != current_state:
                        total_transitions += 1
                    
            return {
                'total_scenarios': total_scenarios,
                'scenario_types': scenario_types,
                'conflict_stats': {
                    'total_conflicts': total_conflicts,
                    'avg_conflicts_per_scenario': total_conflicts / total_scenarios
                },
                'transition_stats': {
                    'total_transitions': total_transitions,
                    'avg_transition_time': 0.0  # TODO: Calculate actual transition time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting scenario stats: {str(e)}")
            return {
                'total_scenarios': 0,
                'scenario_types': {},
                'conflict_stats': {
                    'total_conflicts': 0,
                    'avg_conflicts_per_scenario': 0.0
                },
                'transition_stats': {
                    'total_transitions': 0,
                    'avg_transition_time': 0.0
                }
            }
            
    def _save_scenarios(self) -> None:
        """Save scenarios to file."""
        try:
            # Ensure directory exists
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            scenario_file = self.base_dir / "scenarios.json"
            with open(scenario_file, 'w') as f:
                json.dump(self.scenario_history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving scenarios: {str(e)}")
            
    def _load_scenarios(self) -> None:
        """Load scenarios from file."""
        try:
            scenario_file = self.base_dir / "scenarios.json"
            if scenario_file.exists():
                with open(scenario_file, 'r') as f:
                    self.scenario_history = json.load(f)
                    self.active_scenarios = self.scenario_history[-10:]
            else:
                # Create empty file if it doesn't exist
                with open(scenario_file, 'w') as f:
                    json.dump([], f)
                    
        except Exception as e:
            self.logger.error(f"Error loading scenarios: {str(e)}")
            
    def clear_scenarios(self) -> None:
        """Clear all stored scenarios."""
        try:
            self.scenario_history = []
            self.active_scenarios = []
            self._save_scenarios()
        except Exception as e:
            self.logger.error(f"Error clearing scenarios: {str(e)}")
            
    def get_recent_scenarios(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent scenarios."""
        try:
            return self.active_scenarios[-count:]
        except Exception as e:
            self.logger.error(f"Error getting recent scenarios: {str(e)}")
            return [] 