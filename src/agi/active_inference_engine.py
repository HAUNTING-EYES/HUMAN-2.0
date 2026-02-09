#!/usr/bin/env python3
"""
HUMAN 2.0 Active Inference Engine
================================

Implements active inference and predictive processing for AGI:
- Free energy minimization
- Predictive models of the world
- Action selection based on expected surprise
- Belief updating through prediction error
- Hierarchical message passing

Based on the theoretical framework of Karl Friston and Andy Clark.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class BeliefType(Enum):
    PERCEPTION = "perception"
    ACTION = "action"
    INTENTION = "intention"
    EXPECTATION = "expectation"

@dataclass
class Belief:
    """A belief in the generative model"""
    id: str
    type: BeliefType
    content: Dict[str, Any]
    precision: float  # Inverse variance (confidence)
    timestamp: float
    level: int  # Hierarchical level (0 = sensory, higher = more abstract)

@dataclass
class Observation:
    """An observation from the environment"""
    modality: str
    data: Any
    timestamp: float
    reliability: float = 1.0

@dataclass
class Action:
    """An action to be taken"""
    type: str
    parameters: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    confidence: float
    timestamp: float

class ActiveInferenceEngine:
    """
    Core active inference engine implementing predictive processing
    
    Key principles:
    1. The brain is a prediction machine
    2. Minimize free energy (surprise)
    3. Update beliefs based on prediction error
    4. Select actions to minimize expected future surprise
    """
    
    def __init__(self, learning_rate: float = 0.1, precision_decay: float = 0.95):
        # Core parameters
        self.learning_rate = learning_rate
        self.precision_decay = precision_decay
        
        # Generative model components
        self.beliefs: Dict[str, Belief] = {}
        self.generative_model: Dict[str, Any] = {}
        self.prediction_errors: List[Dict[str, Any]] = []
        
        # Hierarchical levels
        self.num_levels = 5  # From sensory to abstract
        self.level_beliefs: Dict[int, List[str]] = {i: [] for i in range(self.num_levels)}
        
        # Action and perception models
        self.perception_model = self._initialize_perception_model()
        self.action_model = self._initialize_action_model()
        
        # State tracking
        self.current_observations: List[Observation] = []
        self.action_history: List[Action] = []
        self.free_energy_history: List[float] = []
        
        # Consciousness integration
        self.consciousness_interface = None
        
        logger.info("Active Inference Engine initialized")
    
    def _initialize_perception_model(self) -> Dict[str, Any]:
        """Initialize the perception model"""
        return {
            'visual': {
                'emotions': ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral'],
                'objects': ['face', 'person', 'scene'],
                'features': ['brightness', 'contrast', 'color_dominant']
            },
            'text': {
                'emotions': ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral', 'excitement'],
                'sentiment': ['positive', 'negative', 'neutral'],
                'topics': ['emotion', 'conversation', 'question', 'statement']
            },
            'audio': {
                'emotions': ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral'],
                'speech': ['words', 'tone', 'volume', 'pace']
            }
        }
    
    def _initialize_action_model(self) -> Dict[str, Any]:
        """Initialize the action model"""
        return {
            'communication': {
                'respond': {'type': 'text', 'emotional_tone': 'adaptive'},
                'ask_question': {'type': 'curiosity_driven', 'domain': 'adaptive'},
                'express_emotion': {'type': 'physiological_response', 'intensity': 'adaptive'}
            },
            'learning': {
                'update_belief': {'confidence': 'adaptive', 'domain': 'adaptive'},
                'seek_information': {'curiosity_level': 'adaptive', 'priority': 'adaptive'},
                'reflect': {'depth': 'adaptive', 'focus': 'adaptive'}
            },
            'attention': {
                'focus_internal': {'duration': 'adaptive', 'intensity': 'adaptive'},
                'focus_external': {'target': 'adaptive', 'duration': 'adaptive'},
                'switch_attention': {'direction': 'adaptive', 'speed': 'adaptive'}
            }
        }
    
    def process_observation(self, observation: Observation) -> Dict[str, Any]:
        """
        Process a new observation through active inference
        
        1. Generate predictions
        2. Calculate prediction error
        3. Update beliefs
        4. Minimize free energy
        """
        logger.info(f"Processing observation: {observation.modality}")
        
        # Generate predictions based on current beliefs
        predictions = self._generate_predictions(observation.modality)
        
        # Calculate prediction error
        prediction_error = self._calculate_prediction_error(observation, predictions)
        
        # Update beliefs based on prediction error
        self._update_beliefs(prediction_error, observation)
        
        # Calculate free energy
        free_energy = self._calculate_free_energy(prediction_error)
        self.free_energy_history.append(free_energy)
        
        # Store observation
        self.current_observations.append(observation)
        
        # Hierarchical message passing
        self._hierarchical_update(prediction_error, observation)
        
        # Update consciousness systems if available
        if self.consciousness_interface:
            self._update_consciousness(observation, prediction_error, free_energy)
        
        return {
            'prediction_error': prediction_error,
            'free_energy': free_energy,
            'updated_beliefs': len(self.beliefs),
            'predictions': predictions
        }
    
    def _generate_predictions(self, modality: str) -> Dict[str, Any]:
        """Generate predictions based on current beliefs"""
        predictions = {}
        
        # Get relevant beliefs for this modality
        relevant_beliefs = [b for b in self.beliefs.values() 
                          if modality in b.content.get('modalities', [modality])]
        
        if not relevant_beliefs:
            # Default predictions when no beliefs exist
            if modality in self.perception_model:
                predictions = {
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'features': self.perception_model[modality]
                }
            else:
                predictions = {'emotion': 'neutral', 'confidence': 0.1}
        else:
            # Weighted prediction based on belief precision
            total_precision = sum(b.precision for b in relevant_beliefs)
            
            if total_precision > 0:
                predictions = {}
                for belief in relevant_beliefs:
                    weight = belief.precision / total_precision
                    for key, value in belief.content.items():
                        if key not in predictions:
                            predictions[key] = 0
                        if isinstance(value, (int, float)):
                            predictions[key] += value * weight
                        elif isinstance(value, str) and key == 'emotion':
                            # For categorical predictions, use highest precision belief
                            if weight > predictions.get(f'{key}_weight', 0):
                                predictions[key] = value
                                predictions[f'{key}_weight'] = weight
        
        return predictions
    
    def _calculate_prediction_error(self, observation: Observation, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate prediction error between observation and predictions"""
        error = {
            'modality': observation.modality,
            'timestamp': observation.timestamp,
            'errors': {}
        }
        
        # Extract features from observation
        obs_features = self._extract_features(observation)
        
        # Calculate error for each predicted feature
        for feature, predicted_value in predictions.items():
            if feature.endswith('_weight'):
                continue
                
            if feature in obs_features:
                observed_value = obs_features[feature]
                
                if isinstance(predicted_value, (int, float)) and isinstance(observed_value, (int, float)):
                    # Numerical error
                    error['errors'][feature] = abs(predicted_value - observed_value)
                elif isinstance(predicted_value, str) and isinstance(observed_value, str):
                    # Categorical error (0 if match, 1 if mismatch)
                    error['errors'][feature] = 0.0 if predicted_value == observed_value else 1.0
                else:
                    # Default error for mismatched types
                    error['errors'][feature] = 0.5
            else:
                # Missing feature error
                error['errors'][feature] = 0.3
        
        # Calculate overall error magnitude
        if error['errors']:
            error['magnitude'] = np.mean(list(error['errors'].values()))
        else:
            error['magnitude'] = 0.5
        
        # Store prediction error
        self.prediction_errors.append(error)
        
        return error
    
    def _extract_features(self, observation: Observation) -> Dict[str, Any]:
        """Extract features from observation data"""
        features = {}
        
        if observation.modality == 'emotion_result':
            # Emotion recognition result
            if isinstance(observation.data, dict):
                features['emotion'] = observation.data.get('emotion', 'neutral')
                features['confidence'] = observation.data.get('confidence', 0.5)
                features['modality_type'] = observation.data.get('modality', 'unknown')
        
        elif observation.modality == 'text':
            # Text input
            if isinstance(observation.data, str):
                features['content'] = observation.data
                features['length'] = len(observation.data)
                features['has_question'] = '?' in observation.data
                features['emotional_words'] = self._count_emotional_words(observation.data)
        
        elif observation.modality == 'consciousness_state':
            # Consciousness state update
            if isinstance(observation.data, dict):
                features.update(observation.data)
        
        return features
    
    def _count_emotional_words(self, text: str) -> int:
        """Count emotional words in text"""
        emotional_words = [
            'happy', 'sad', 'angry', 'afraid', 'surprised', 'excited',
            'joy', 'fear', 'love', 'hate', 'wonderful', 'terrible',
            'amazing', 'awful', 'fantastic', 'horrible'
        ]
        
        text_lower = text.lower()
        return sum(1 for word in emotional_words if word in text_lower)
    
    def _update_beliefs(self, prediction_error: Dict[str, Any], observation: Observation):
        """Update beliefs based on prediction error"""
        
        # Create or update belief based on observation
        belief_id = f"{observation.modality}_{int(observation.timestamp)}"
        
        # Calculate precision based on prediction error and observation reliability
        error_magnitude = prediction_error['magnitude']
        precision = observation.reliability * (1.0 - error_magnitude)
        precision = max(0.1, min(1.0, precision))  # Clamp between 0.1 and 1.0
        
        # Determine hierarchical level
        level = self._determine_belief_level(observation.modality)
        
        # Create new belief
        new_belief = Belief(
            id=belief_id,
            type=BeliefType.PERCEPTION,
            content={
                'modalities': [observation.modality],
                'features': self._extract_features(observation),
                'prediction_error': error_magnitude,
                'source': 'observation'
            },
            precision=precision,
            timestamp=observation.timestamp,
            level=level
        )
        
        # Store belief
        self.beliefs[belief_id] = new_belief
        self.level_beliefs[level].append(belief_id)
        
        # Update related beliefs (spreading activation)
        self._update_related_beliefs(new_belief, prediction_error)
        
        # Decay precision of old beliefs
        self._decay_belief_precision()
    
    def _determine_belief_level(self, modality: str) -> int:
        """Determine hierarchical level for belief"""
        level_map = {
            'visual': 0,  # Sensory level
            'audio': 0,
            'text': 1,    # Linguistic level
            'emotion_result': 2,  # Emotional level
            'consciousness_state': 3,  # Meta-cognitive level
            'reflection': 4   # Abstract level
        }
        return level_map.get(modality, 1)
    
    def _update_related_beliefs(self, new_belief: Belief, prediction_error: Dict[str, Any]):
        """Update beliefs related to the new belief"""
        for belief_id, belief in self.beliefs.items():
            if belief_id == new_belief.id:
                continue
            
            # Check for semantic similarity
            similarity = self._calculate_belief_similarity(new_belief, belief)
            
            if similarity > 0.3:  # Threshold for updating related beliefs
                # Update precision based on prediction error
                error_impact = prediction_error['magnitude'] * similarity
                belief.precision *= (1.0 - error_impact * self.learning_rate)
                belief.precision = max(0.1, belief.precision)
    
    def _calculate_belief_similarity(self, belief1: Belief, belief2: Belief) -> float:
        """Calculate similarity between two beliefs"""
        similarity = 0.0
        
        # Level similarity
        level_diff = abs(belief1.level - belief2.level)
        level_similarity = 1.0 / (1.0 + level_diff)
        similarity += level_similarity * 0.3
        
        # Modality similarity
        modalities1 = set(belief1.content.get('modalities', []))
        modalities2 = set(belief2.content.get('modalities', []))
        
        if modalities1 and modalities2:
            modality_overlap = len(modalities1.intersection(modalities2))
            modality_total = len(modalities1.union(modalities2))
            modality_similarity = modality_overlap / modality_total if modality_total > 0 else 0
            similarity += modality_similarity * 0.4
        
        # Feature similarity
        features1 = belief1.content.get('features', {})
        features2 = belief2.content.get('features', {})
        
        if features1 and features2:
            common_features = set(features1.keys()).intersection(set(features2.keys()))
            feature_similarity = 0.0
            
            for feature in common_features:
                val1, val2 = features1[feature], features2[feature]
                if isinstance(val1, str) and isinstance(val2, str):
                    feature_similarity += 1.0 if val1 == val2 else 0.0
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    feature_similarity += 1.0 - abs(val1 - val2)
            
            if common_features:
                feature_similarity /= len(common_features)
                similarity += feature_similarity * 0.3
        
        return min(1.0, similarity)
    
    def _decay_belief_precision(self):
        """Decay precision of beliefs over time"""
        current_time = time.time()
        
        for belief in self.beliefs.values():
            age = current_time - belief.timestamp
            decay_factor = self.precision_decay ** (age / 60.0)  # Decay per minute
            belief.precision *= decay_factor
            belief.precision = max(0.01, belief.precision)  # Minimum precision
    
    def _calculate_free_energy(self, prediction_error: Dict[str, Any]) -> float:
        """
        Calculate free energy (surprise)
        F = E_q[ln q(μ) - ln p(o,μ)]
        
        Simplified as weighted prediction error
        """
        error_magnitude = prediction_error['magnitude']
        
        # Weight by current belief precision
        total_precision = sum(b.precision for b in self.beliefs.values())
        precision_weight = 1.0 / (1.0 + total_precision)
        
        # Free energy is higher when:
        # 1. Prediction error is high
        # 2. Current beliefs are uncertain (low precision)
        free_energy = error_magnitude * (1.0 + precision_weight)
        
        return free_energy
    
    def _hierarchical_update(self, prediction_error: Dict[str, Any], observation: Observation):
        """Perform hierarchical message passing"""
        error_magnitude = prediction_error['magnitude']
        obs_level = self._determine_belief_level(observation.modality)
        
        # Pass prediction error up the hierarchy
        for level in range(obs_level + 1, self.num_levels):
            level_beliefs = [self.beliefs[bid] for bid in self.level_beliefs[level] 
                           if bid in self.beliefs]
            
            for belief in level_beliefs:
                # Higher-level beliefs get updated based on lower-level errors
                update_strength = error_magnitude * (0.5 ** (level - obs_level))
                belief.precision *= (1.0 - update_strength * self.learning_rate)
                belief.precision = max(0.1, belief.precision)
        
        # Pass predictions down the hierarchy
        for level in range(obs_level - 1, -1, -1):
            level_beliefs = [self.beliefs[bid] for bid in self.level_beliefs[level] 
                           if bid in self.beliefs]
            
            for belief in level_beliefs:
                # Lower-level beliefs get refined by higher-level predictions
                refinement = (1.0 - error_magnitude) * (0.8 ** (obs_level - level))
                belief.precision *= (1.0 + refinement * self.learning_rate * 0.5)
                belief.precision = min(1.0, belief.precision)
    
    def select_action(self) -> Optional[Action]:
        """
        Select action to minimize expected free energy
        
        Actions are selected to:
        1. Reduce uncertainty (epistemic value)
        2. Achieve preferred outcomes (pragmatic value)
        """
        if not self.beliefs:
            return None
        
        # Calculate current uncertainty
        current_uncertainty = self._calculate_uncertainty()
        
        # Generate possible actions
        possible_actions = self._generate_possible_actions()
        
        if not possible_actions:
            return None
        
        # Evaluate each action
        best_action = None
        best_score = float('inf')
        
        for action in possible_actions:
            expected_free_energy = self._calculate_expected_free_energy(action)
            
            if expected_free_energy < best_score:
                best_score = expected_free_energy
                best_action = action
        
        if best_action:
            self.action_history.append(best_action)
            logger.info(f"Selected action: {best_action.type}")
        
        return best_action
    
    def _calculate_uncertainty(self) -> float:
        """Calculate current epistemic uncertainty"""
        if not self.beliefs:
            return 1.0
        
        # Uncertainty is inverse of average precision
        avg_precision = np.mean([b.precision for b in self.beliefs.values()])
        uncertainty = 1.0 / (1.0 + avg_precision)
        
        return uncertainty
    
    def _generate_possible_actions(self) -> List[Action]:
        """Generate possible actions based on current state"""
        actions = []
        current_time = time.time()
        
        # Communication actions
        if self._should_communicate():
            actions.append(Action(
                type='communicate',
                parameters={'method': 'respond', 'emotional_tone': 'adaptive'},
                expected_outcome={'engagement': 0.7, 'information_gain': 0.5},
                confidence=0.6,
                timestamp=current_time
            ))
        
        # Learning actions
        if self._should_seek_information():
            actions.append(Action(
                type='seek_information',
                parameters={'domain': 'high_uncertainty_area', 'method': 'question'},
                expected_outcome={'uncertainty_reduction': 0.8, 'knowledge_gain': 0.7},
                confidence=0.7,
                timestamp=current_time
            ))
        
        # Reflection actions
        if self._should_reflect():
            actions.append(Action(
                type='reflect',
                parameters={'focus': 'recent_experiences', 'depth': 'moderate'},
                expected_outcome={'insight_generation': 0.6, 'belief_refinement': 0.5},
                confidence=0.5,
                timestamp=current_time
            ))
        
        return actions
    
    def _should_communicate(self) -> bool:
        """Determine if communication action is warranted"""
        # Communicate if recent observations suggest interaction
        recent_obs = [obs for obs in self.current_observations 
                     if time.time() - obs.timestamp < 30]
        
        return any(obs.modality in ['text', 'conversation'] for obs in recent_obs)
    
    def _should_seek_information(self) -> bool:
        """Determine if information seeking is warranted"""
        uncertainty = self._calculate_uncertainty()
        return uncertainty > 0.6
    
    def _should_reflect(self) -> bool:
        """Determine if reflection is warranted"""
        # Reflect if there have been significant prediction errors
        recent_errors = [err for err in self.prediction_errors 
                        if time.time() - err['timestamp'] < 60]
        
        if recent_errors:
            avg_error = np.mean([err['magnitude'] for err in recent_errors])
            return avg_error > 0.5
        
        return False
    
    def _calculate_expected_free_energy(self, action: Action) -> float:
        """Calculate expected free energy for an action"""
        # Simplified expected free energy calculation
        
        # Epistemic value (uncertainty reduction)
        epistemic_value = action.expected_outcome.get('uncertainty_reduction', 0.0)
        epistemic_value += action.expected_outcome.get('information_gain', 0.0)
        
        # Pragmatic value (goal achievement)
        pragmatic_value = action.expected_outcome.get('engagement', 0.0)
        pragmatic_value += action.expected_outcome.get('insight_generation', 0.0)
        
        # Cost of action (inverse of confidence)
        action_cost = 1.0 - action.confidence
        
        # Expected free energy (lower is better)
        expected_free_energy = action_cost - (epistemic_value + pragmatic_value) * 0.5
        
        return expected_free_energy
    
    def _update_consciousness(self, observation: Observation, prediction_error: Dict[str, Any], free_energy: float):
        """Update consciousness systems with active inference results"""
        if not self.consciousness_interface:
            return
        
        try:
            # Update self-awareness with prediction and error
            self.consciousness_interface['self_awareness'].add_experience({
                'type': 'active_inference',
                'content': f"Processed {observation.modality} with {prediction_error['magnitude']:.2f} error",
                'prediction_error': prediction_error['magnitude'],
                'free_energy': free_energy,
                'beliefs_updated': len(self.beliefs)
            })
            
            # Update curiosity based on prediction error
            if prediction_error['magnitude'] > 0.5:
                self.consciousness_interface['curiosity'].update_knowledge({
                    'concept': f'prediction_error_{observation.modality}',
                    'error_magnitude': prediction_error['magnitude'],
                    'related_concepts': ['active_inference', 'learning', observation.modality]
                })
            
            # Update physiology based on free energy
            emotion_intensity = min(1.0, free_energy)
            if free_energy > 0.7:
                # High free energy = surprise/uncertainty
                self.consciousness_interface['physiology'].update({
                    'surprise': emotion_intensity,
                    'curiosity': emotion_intensity * 0.8
                })
            else:
                # Low free energy = confidence/satisfaction
                self.consciousness_interface['physiology'].update({
                    'satisfaction': 1.0 - emotion_intensity,
                    'confidence': 1.0 - emotion_intensity
                })
                
        except Exception as e:
            logger.error(f"Consciousness update failed: {e}")
    
    def set_consciousness_interface(self, consciousness_systems: Dict[str, Any]):
        """Set consciousness systems interface"""
        self.consciousness_interface = consciousness_systems
        logger.info("Consciousness interface connected to Active Inference Engine")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current active inference state"""
        current_uncertainty = self._calculate_uncertainty()
        recent_free_energy = self.free_energy_history[-10:] if self.free_energy_history else [0.5]
        
        return {
            'beliefs': len(self.beliefs),
            'uncertainty': current_uncertainty,
            'recent_free_energy': recent_free_energy,
            'avg_free_energy': np.mean(recent_free_energy),
            'prediction_errors': len(self.prediction_errors),
            'actions_taken': len(self.action_history),
            'hierarchical_levels': {
                level: len(beliefs) for level, beliefs in self.level_beliefs.items()
            }
        }
    
    def get_insights(self) -> List[str]:
        """Get insights from active inference processing"""
        insights = []
        
        if self.free_energy_history:
            avg_free_energy = np.mean(self.free_energy_history[-10:])
            if avg_free_energy > 0.7:
                insights.append("High free energy detected - experiencing significant surprise/uncertainty")
            elif avg_free_energy < 0.3:
                insights.append("Low free energy - predictions are accurate and world model is stable")
        
        if self.prediction_errors:
            recent_errors = [err['magnitude'] for err in self.prediction_errors[-5:]]
            if recent_errors:
                avg_error = np.mean(recent_errors)
                if avg_error > 0.6:
                    insights.append("High prediction errors - world model needs updating")
                elif avg_error < 0.2:
                    insights.append("Low prediction errors - world model is well-calibrated")
        
        uncertainty = self._calculate_uncertainty()
        if uncertainty > 0.8:
            insights.append("High epistemic uncertainty - need more information")
        elif uncertainty < 0.2:
            insights.append("Low epistemic uncertainty - confident in current beliefs")
        
        return insights

# Factory function
def create_active_inference_engine(learning_rate: float = 0.1) -> ActiveInferenceEngine:
    """Create and initialize active inference engine"""
    return ActiveInferenceEngine(learning_rate=learning_rate) 