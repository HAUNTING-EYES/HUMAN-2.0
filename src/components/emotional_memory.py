from typing import Dict, List, Any, Optional, Tuple
import logging
import datetime
import time
import numpy as np
from pathlib import Path
import json
import random
import math
import os
import glob
import tempfile

class EmotionalMemoryCore:
    """Core component for managing emotional state."""
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize EmotionalMemoryCore component."""
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set up directories
        if base_dir is None:
            base_dir = tempfile.mkdtemp()
        self.base_dir = Path(base_dir)
        self.memory_dir = self.base_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize emotional dimensions with ranges
        self.emotion_dimensions = {
            'valence': {'min': -1.0, 'max': 1.0, 'default': 0.0},
            'arousal': {'min': 0.0, 'max': 1.0, 'default': 0.5},
            'dominance': {'min': 0.0, 'max': 1.0, 'default': 0.5},
            'novelty': {'min': 0.0, 'max': 1.0, 'default': 0.5},
            'complexity': {'min': 0.0, 'max': 1.0, 'default': 0.5},
            'intensity': {'min': 0.0, 'max': 1.0, 'default': 0.5},
            'stability': {'min': 0.0, 'max': 1.0, 'default': 0.5},
            'coherence': {'min': 0.0, 'max': 1.0, 'default': 0.5}
        }
        
        # Initialize emotional state
        self.emotional_state = {
            dim: config['default'] 
            for dim, config in self.emotion_dimensions.items()
        }
        self.emotional_state['timestamp'] = datetime.datetime.now().isoformat()
        
        # Initialize personality traits
        self.personality_traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        
        # Initialize emotional inertia
        self.emotional_inertia = {
            dim: 0.5 for dim in self.emotion_dimensions.keys()
        }
        
        # Initialize emergence patterns
        self.emergence_patterns = self._initialize_emergence_patterns()
        
        # Initialize memory buffers
        self.short_term_memory = []
        self.long_term_memory = []
        self.working_memory = []
        
        try:
            self._load_memories()
        except Exception as e:
            self.logger.error(f"Failed to load memories: {e}")
            self.short_term_memory = []
            self.long_term_memory = []
            self.working_memory = []
    
    def _load_memories(self) -> None:
        """Load memories from disk if they exist."""
        try:
            short_term_path = self.memory_dir / "short_term.json"
            long_term_path = self.memory_dir / "long_term.json"
            
            if short_term_path.exists():
                with open(short_term_path) as f:
                    self.short_term_memory = json.load(f)
            
            if long_term_path.exists():
                with open(long_term_path) as f:
                    self.long_term_memory = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading memories: {e}")
            raise
    
    def process_emotional_state(self, new_state: Dict[str, float]) -> Dict[str, Any]:
        """Process and validate a new emotional state."""
        try:
            processed_state = {}
            
            for dim, config in self.emotion_dimensions.items():
                if dim in new_state:
                    value = new_state[dim]
                    min_val = config['min']
                    max_val = config['max']
                    processed_state[dim] = max(min_val, min(max_val, value))
                else:
                    processed_state[dim] = config['default']
            
            processed_state['timestamp'] = datetime.datetime.now().isoformat()
            
            return {
                'state': processed_state,
                'style': self._calculate_emotional_style(),
                'patterns': self.detect_emergence_patterns(processed_state),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing emotional state: {e}")
            return {
                'state': self.emotional_state.copy(),
                'style': {},
                'patterns': [],
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        try:
            # Simple rule-based sentiment analysis
            positive_words = {
                'good', 'great', 'happy', 'wonderful', 'excellent', 'amazing', 'love', 'beautiful',
                'joy', 'excited', 'delighted', 'fantastic', 'pleased', 'glad', 'positive', 'awesome'
            }
            negative_words = {
                'bad', 'terrible', 'sad', 'awful', 'horrible', 'hate', 'ugly', 'worst',
                'angry', 'upset', 'disappointed', 'frustrated', 'negative', 'annoyed', 'miserable', 'depressed'
            }
            
            # Intensity modifiers
            intensifiers = {
                'very': 1.5,
                'really': 1.5,
                'extremely': 2.0,
                'incredibly': 2.0,
                'somewhat': 0.5,
                'slightly': 0.5
            }
            
            words = text.lower().split()
            
            # Count sentiment words with intensity
            pos_score = 0.0
            neg_score = 0.0
            current_intensity = 1.0
            
            for i, word in enumerate(words):
                # Check for intensifiers
                if word in intensifiers:
                    current_intensity = intensifiers[word]
                    continue
                    
                # Apply sentiment scoring
                if word in positive_words:
                    pos_score += current_intensity
                elif word in negative_words:
                    neg_score += current_intensity
                    
                # Reset intensity after use
                current_intensity = 1.0
            
            # Calculate final sentiment
            if pos_score == 0 and neg_score == 0:
                return -0.2  # Slight negative bias for neutral text
                
            total = pos_score + neg_score
            if total == 0:
                return 0.0
                
            return (pos_score - neg_score) / total
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0
    
    def process_interaction(self, input_text: str) -> Dict[str, Any]:
        """Process an interaction and generate a response.
        
        Args:
            input_text: Input text to process
            
        Returns:
            Dict[str, Any]: Processing result including response and sentiment
        """
        try:
            # Generate emotional response
            response_data = self.generate_emotional_response(input_text)
            
            # Create memory entry
            memory_entry = {
                'text': input_text,
                'sentiment': response_data['sentiment'],
                'response': response_data['response'],
                'emotional_state': self.emotional_state.copy(),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Store in memory buffers
            self.short_term_memory.append(memory_entry)
            if len(self.short_term_memory) > 10:
                self.long_term_memory.append(self.short_term_memory.pop(0))
                
            return {
                'response': response_data['response'],
                'sentiment_score': response_data['sentiment'],
                'emotional_state': response_data['emotional_state'],
                'memory_entry': memory_entry
            }
            
        except Exception as e:
            self.logger.error(f"Error processing interaction: {str(e)}")
            return {
                'response': "I'm processing that.",
                'sentiment_score': 0.0,
                'emotional_state': self.emotional_state.copy(),
                'memory_entry': None
            }
    
    def process_emotion(self, emotion_type: str, intensity: float = 0.5, description: str = "", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process an emotion and update emotional state.
        
        Args:
            emotion_type: Type of emotion (e.g., 'joy', 'sadness')
            intensity: Intensity of the emotion (0 to 1)
            description: Optional description of the emotional event
            context: Optional context dictionary
            
        Returns:
            Dict[str, Any]: Processing result with success flag
        """
        try:
            # Define emotion mappings
            emotion_mappings = {
                'joy': {'valence': 0.8, 'arousal': 0.6, 'dominance': 0.7},
                'sadness': {'valence': -0.6, 'arousal': -0.4, 'dominance': -0.3},
                'anger': {'valence': -0.7, 'arousal': 0.8, 'dominance': 0.6},
                'fear': {'valence': -0.8, 'arousal': 0.7, 'dominance': -0.6},
                'surprise': {'valence': 0.2, 'arousal': 0.8, 'dominance': 0.0},
                'disgust': {'valence': -0.6, 'arousal': 0.5, 'dominance': 0.4}
            }
            
            if emotion_type not in emotion_mappings:
                raise ValueError(f"Unknown emotion type: {emotion_type}")
                
            # Get emotion values
            emotion_values = emotion_mappings[emotion_type]
            
            # Update emotional state with intensity scaling
            new_state = {}
            for dim, value in emotion_values.items():
                scaled_value = value * intensity
                current = self.emotional_state.get(dim, 0.5)
                new_state[dim] = self._apply_emotional_inertia(current + scaled_value, dim)
                
            self.update_emotional_state(new_state)
            
            return {
                'success': True,
                'emotion': emotion_type,
                'intensity': intensity,
                'description': description,
                'emotional_state': self.emotional_state.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing emotion: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_emotional_response(self, input_text: str, emotional_context: Dict[str, float] = None, personality_context: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate an emotional response based on input text and context.
        
        Args:
            input_text: Input text to respond to
            emotional_context: Optional emotional context dictionary
            personality_context: Optional personality context dictionary
            
        Returns:
            Dict[str, Any]: Response data including text and sentiment
        """
        try:
            # Analyze sentiment
            sentiment = self._analyze_sentiment(input_text)
            
            # Get emotional context
            context = emotional_context if emotional_context else self.emotional_state
            
            # Select response template
            templates = self._get_response_templates(sentiment)
            template = self._select_template(templates, context)
            
            # Generate base response
            base_response = self._generate_base_response(template, input_text, sentiment)
            
            # Get emotional style and preferences
            style = self._get_emotional_style()
            preferences = self._get_interaction_preferences()
            
            # Modify response based on style and preferences
            modified_response = self._modify_response(base_response, style, preferences)
            
            # Update emotional state based on interaction
            self.process_emotion('joy' if sentiment > 0 else 'sadness', abs(sentiment))
            
            return {
                'response': modified_response,
                'sentiment': sentiment,
                'style': style,
                'preferences': preferences,
                'emotional_state': self.emotional_state.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating emotional response: {str(e)}")
            return {
                'response': "I'm processing that.",
                'sentiment': 0.0,
                'style': self._get_emotional_style(),
                'preferences': self._get_interaction_preferences(),
                'emotional_state': self.emotional_state.copy()
            }
    
    def _get_response_templates(self, sentiment_score: float) -> List[str]:
        """Get response templates based on sentiment score.
        
        Args:
            sentiment_score: Sentiment score of input text
            
        Returns:
            List[str]: List of response templates
        """
        try:
            # Define templates for different sentiment ranges
            positive_templates = [
                "That's wonderful! {}",
                "I'm so happy to hear that! {}",
                "That's great news! {}",
                "How delightful! {}"
            ]
            
            negative_templates = [
                "I'm sorry to hear that. {}",
                "That must be difficult. {}",
                "I understand how you feel. {}",
                "I'm here to listen. {}"
            ]
            
            neutral_templates = [
                "I see. {}",
                "Interesting. {}",
                "Tell me more about that. {}",
                "I'm listening. {}"
            ]
            
            # Select templates based on sentiment
            if sentiment_score > 0.3:
                return positive_templates
            elif sentiment_score < -0.3:
                return negative_templates
            else:
                return neutral_templates
                
        except Exception as e:
            self.logger.error(f"Error getting response templates: {str(e)}")
            return ["I understand. {}"]
    
    def _select_template(self, templates: List[str], emotional_context: Dict[str, float]) -> str:
        """Select a response template based on emotional context.
        
        Args:
            templates: List of response templates
            emotional_context: Emotional context dictionary
            
        Returns:
            str: Selected template
        """
        try:
            # Get emotional style
            style = self._get_emotional_style()
            expressiveness = style['expressiveness']
            empathy = style['empathy']
            
            # Calculate weights for each template
            weights = []
            for template in templates:
                weight = 1.0
                
                # Adjust weight based on template content and emotional context
                if "!" in template:
                    weight *= expressiveness
                if "understand" in template.lower() or "sorry" in template.lower():
                    weight *= empathy
                if "interesting" in template.lower():
                    weight *= emotional_context.get('novelty', 0.5)
                if "tell me more" in template.lower():
                    weight *= emotional_context.get('complexity', 0.5)
                
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(templates)] * len(templates)
            
            # Select template based on weights
            return random.choices(templates, weights=weights)[0]
            
        except Exception as e:
            self.logger.error(f"Error selecting template: {e}")
            return templates[0] if templates else "I understand."
    
    def _get_emotional_style(self) -> Dict[str, float]:
        """Get current emotional style based on personality traits and state.
        
        Returns:
            Dict[str, float]: Emotional style parameters
        """
        try:
            # Calculate expressiveness from personality and emotional state
            expressiveness = (
                0.4 * self.personality_traits['extraversion'] +
                0.3 * self.emotional_state.get('arousal', 0.5) +
                0.3 * self.emotional_state.get('valence', 0.5)
            )
            
            # Calculate empathy from personality
            empathy = (
                0.4 * self.personality_traits['agreeableness'] +
                0.3 * (1 - self.personality_traits['neuroticism']) +
                0.3 * self.emotional_state.get('dominance', 0.5)
            )
            
            return {
                'expressiveness': max(0.0, min(1.0, expressiveness)),
                'empathy': max(0.0, min(1.0, empathy))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting emotional style: {str(e)}")
            return {
                'expressiveness': 0.5,
                'empathy': 0.5
            }
    
    def _get_interaction_preferences(self) -> Dict[str, float]:
        """Get interaction preferences based on personality and state.
        
        Returns:
            Dict[str, float]: Interaction preference parameters
        """
        try:
            # Calculate emotional expression preference
            emotional_expression = (
                0.4 * self.personality_traits['extraversion'] +
                0.3 * self.personality_traits['openness'] +
                0.3 * (1 - self.personality_traits['neuroticism'])
            )
            
            # Calculate boundaries preference
            boundaries = (
                0.4 * self.personality_traits['conscientiousness'] +
                0.3 * (1 - self.personality_traits['agreeableness']) +
                0.3 * self.emotional_state.get('dominance', 0.5)
            )
            
            return {
                'emotional_expression': max(0.0, min(1.0, emotional_expression)),
                'boundaries': max(0.0, min(1.0, boundaries))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting interaction preferences: {str(e)}")
            return {
                'emotional_expression': 0.5,
                'boundaries': 0.5
            }
    
    def _calculate_emotional_intelligence(self) -> float:
        """Calculate emotional intelligence score based on personality traits.
        
        Returns:
            float: Emotional intelligence score between 0 and 1
        """
        try:
            # Calculate emotional intelligence from personality traits
            emotional_intelligence = (
                0.3 * self.personality_traits['openness'] +
                0.3 * self.personality_traits['agreeableness'] +
                0.2 * (1 - self.personality_traits['neuroticism']) +
                0.2 * self.personality_traits['conscientiousness']
            )
            return max(0.0, min(1.0, emotional_intelligence))
        except Exception as e:
            self.logger.error(f"Error calculating emotional intelligence: {str(e)}")
            return 0.5
    
    def _generate_base_response(self, template: str, input_text: str, sentiment: float) -> str:
        """Generate base response from template.
        
        Args:
            template: Response template to use
            input_text: Input text to respond to
            sentiment: Sentiment score of input text
            
        Returns:
            str: Generated base response
        """
        try:
            # Extract key phrases from input
            words = input_text.lower().split()
            key_phrases = [word for word in words if len(word) > 3]
            context = ' '.join(key_phrases) if key_phrases else input_text
            
            # Format template with context
            return template.format(context)
            
        except Exception as e:
            self.logger.error(f"Error generating base response: {str(e)}")
            return "I understand."
    
    def _modify_response(self, base_response: str, style: Dict[str, Any], preferences: Dict[str, Any]) -> str:
        """Modify response based on emotional style and preferences.
        
        Args:
            base_response: The base response to modify
            style: Emotional style dictionary
            preferences: Interaction preferences dictionary
            
        Returns:
            str: Modified response
        """
        try:
            # Get style parameters
            expressiveness = float(style.get('expressiveness', 0.5))
            empathy = float(style.get('empathy', 0.5))
            
            # Get preference parameters
            emotional_expression = float(preferences.get('emotional_expression', 0.5))
            boundaries = float(preferences.get('boundaries', 0.5))
            
            # Calculate modification factors
            expression_factor = (expressiveness + emotional_expression) / 2
            empathy_factor = (empathy + (1 - boundaries)) / 2
            
            # Apply modifications based on factors
            if expression_factor > 0.7:
                base_response = base_response.upper()
            elif expression_factor < 0.3:
                base_response = base_response.lower()
                
            if empathy_factor > 0.7:
                base_response = f"I understand. {base_response}"
            elif empathy_factor < 0.3:
                base_response = f"Let me process that. {base_response}"
                
            return base_response
            
        except Exception as e:
            self.logger.error(f"Error modifying response: {str(e)}")
            return base_response
    
    def _initialize_emergence_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize emergence patterns structure.
        
        Returns:
            Dict[str, Dict[str, Any]]: Initialized emergence patterns
        """
        return {
            'emotional_resonance': {
                'type': 'resonance',
                'strength': 0.3,
                'conditions': {
                    'valence': {'min': 0.6},
                    'arousal': {'min': 0.5}
                }
            },
            'emotional_contagion': {
                'type': 'contagion',
                'strength': 0.4,
                'conditions': {
                    'arousal': {'min': 0.4},
                    'valence': {'range': [-0.8, 0.8]}
                }
            },
            'emotional_regulation': {
                'type': 'regulation',
                'strength': 0.5,
                'conditions': {
                    'arousal': {'max': 0.7},
                    'dominance': {'min': 0.4}
                }
            },
            'emotional_adaptation': {
                'type': 'adaptation',
                'strength': 0.6,
                'conditions': {
                    'valence': {'range': [-0.6, 0.6]},
                    'arousal': {'range': [0.3, 0.7]}
                }
            }
        }
    
    def detect_emergence_patterns(self, emotional_state: Dict[str, float]) -> List[str]:
        """Detect emergence patterns in emotional state.
        
        Args:
            emotional_state: Current emotional state
            
        Returns:
            List[str]: List of detected patterns
        """
        try:
            detected_patterns = []
            
            for pattern_name, pattern_info in self.emergence_patterns.items():
                conditions = pattern_info['conditions']
                matches = True
                
                for dimension, condition in conditions.items():
                    if dimension not in emotional_state:
                        matches = False
                        break
                        
                    value = emotional_state[dimension]
                    
                    if 'min' in condition and value < condition['min']:
                        matches = False
                        break
                    if 'max' in condition and value > condition['max']:
                        matches = False
                        break
                    if 'range' in condition:
                        min_val, max_val = condition['range']
                        if value < min_val or value > max_val:
                            matches = False
                            break
                
                if matches:
                    detected_patterns.append(pattern_name)
            
            return detected_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting emergence patterns: {str(e)}")
            return []
    
    def _apply_emotional_inertia(self, current_val: float, dimension: str) -> float:
        """Apply emotional inertia to a value.
        
        Args:
            current_val: Current value to apply inertia to
            dimension: Emotional dimension name
            
        Returns:
            float: Value with inertia applied
        """
        try:
            # Get inertia factor for dimension
            inertia = float(self.emotional_inertia.get(dimension, 0.5))
            
            # Get default value for dimension
            default_val = float(self.emotion_dimensions[dimension]['default'])
            
            # Apply inertia
            new_val = current_val * inertia + default_val * (1 - inertia)
            
            # Clamp to valid range
            min_val = float(self.emotion_dimensions[dimension]['min'])
            max_val = float(self.emotion_dimensions[dimension]['max'])
            return max(min_val, min(max_val, new_val))
            
        except Exception as e:
            self.logger.error(f"Error applying emotional inertia: {str(e)}")
            return current_val
    
    def get_personality_profile(self) -> Dict[str, Any]:
        """Get current personality profile.
        
        Returns:
            Dict[str, Any]: Personality profile data
        """
        try:
            return {
                'traits': self.personality_traits.copy(),
                'adaptability': self._calculate_adaptability(),
                'resilience': self._calculate_resilience(),
                'emotional_intelligence': self._calculate_emotional_intelligence(),
                'emotional_style': self._get_emotional_style(),
                'interaction_preferences': self._get_interaction_preferences(),
                'timestamp': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting personality profile: {str(e)}")
            return {
                'traits': self.personality_traits.copy(),
                'adaptability': 0.5,
                'resilience': 0.5,
                'emotional_intelligence': 0.5,
                'emotional_style': {'expressiveness': 0.5, 'empathy': 0.5},
                'interaction_preferences': {'emotional_expression': 0.5, 'boundaries': 0.5},
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def _calculate_adaptability(self) -> float:
        """Calculate adaptability score based on personality traits and emotional state.
        
        Returns:
            float: Adaptability score between 0 and 1
        """
        try:
            # Calculate adaptability from personality traits and emotional flexibility
            adaptability = (
                0.3 * self.personality_traits['openness'] +
                0.3 * (1 - self.personality_traits['neuroticism']) +
                0.2 * self.emotional_state.get('stability', 0.5) +
                0.2 * self.emotional_state.get('coherence', 0.5)
            )
            return max(0.0, min(1.0, adaptability))
        except Exception as e:
            self.logger.error(f"Error calculating adaptability: {str(e)}")
            return 0.5
    
    def _calculate_resilience(self) -> float:
        """Calculate resilience score based on personality traits and emotional state.
        
        Returns:
            float: Resilience score between 0 and 1
        """
        try:
            # Calculate resilience from personality traits and emotional strength
            resilience = (
                0.3 * self.personality_traits['conscientiousness'] +
                0.3 * (1 - self.personality_traits['neuroticism']) +
                0.2 * self.emotional_state.get('stability', 0.5) +
                0.2 * self.emotional_state.get('dominance', 0.5)
            )
            return max(0.0, min(1.0, resilience))
        except Exception as e:
            self.logger.error(f"Error calculating resilience: {str(e)}")
            return 0.5
    
    def initialize(self) -> bool:
        """Initialize the emotional memory system.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing EmotionalMemory system...")
            
            # Ensure memory directories exist
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing memories if available
            self._load_memories()
            
            # Initialize memory attributes if not already set
            if not hasattr(self, 'memories'):
                self.memories = self.short_term_memory + self.long_term_memory
            
            self.logger.info("EmotionalMemory system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize EmotionalMemory: {str(e)}")
            return False
    
    def add_experience(self, experience: Dict[str, Any]) -> bool:
        """Add an experience to emotional memory.
        
        Args:
            experience: Experience data to store
            
        Returns:
            bool: True if experience added successfully
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in experience:
                experience['timestamp'] = datetime.datetime.now().isoformat()
            
            # Add to short-term memory
            self.short_term_memory.append(experience)
            
            # Move to long-term if short-term is full
            if len(self.short_term_memory) > 10:
                self.long_term_memory.append(self.short_term_memory.pop(0))
            
            # Update memories list
            if not hasattr(self, 'memories'):
                self.memories = []
            self.memories = self.short_term_memory + self.long_term_memory
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add experience: {str(e)}")
            return False
    
    def update_emotional_state(self, new_state: Dict[str, float]) -> bool:
        """Update the current emotional state.
        
        Args:
            new_state: New emotional state values
            
        Returns:
            bool: True if update successful
        """
        try:
            # Validate and update each dimension
            for dim, value in new_state.items():
                if dim in self.emotion_dimensions:
                    min_val = self.emotion_dimensions[dim]['min']
                    max_val = self.emotion_dimensions[dim]['max']
                    self.emotional_state[dim] = max(min_val, min(max_val, value))
            
            # Update timestamp
            self.emotional_state['timestamp'] = datetime.datetime.now().isoformat()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update emotional state: {str(e)}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup the emotional memory system.
        
        Returns:
            bool: True if cleanup successful
        """
        try:
            self.logger.info("Cleaning up EmotionalMemory system...")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup EmotionalMemory: {str(e)}")
            return False 