"""
HUMAN 2.0 - Artificial General Intelligence System

This package contains the core components of HUMAN 2.0, including:
- Core cognitive systems
- Consciousness frameworks
- Emotional intelligence systems
- Advanced AI components
- Training and inference modules
"""

__version__ = "2.0.0"
__author__ = "HUMAN 2.0 Development Team"

# Core imports
from .core.human import HUMAN
from .consciousness.self_awareness import SelfAwarenessSystem
from .consciousness.reflection import ReflectionEngine
from .consciousness.curiosity import CuriosityEngine
from .consciousness.physiology import PhysiologicalSystem

# Component imports
from .components.multimodal_emotion_processor import MultimodalEmotionProcessor
from .components.emotional_memory import EmotionalMemoryCore as EmotionalMemory
from .components.emotional_learning import EmotionalLearningSystem

__all__ = [
    'HUMAN',
    'SelfAwarenessSystem',
    'ReflectionEngine', 
    'CuriosityEngine',
    'PhysiologicalSystem',
    'MultimodalEmotionProcessor',
    'EmotionalMemory',
    'EmotionalLearningSystem'
] 