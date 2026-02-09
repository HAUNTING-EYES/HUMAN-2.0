"""
Core systems for HUMAN 2.0

This package contains the fundamental cognitive and system components:
- Main HUMAN class
- Configuration management
- Pattern recognition
- Resource monitoring
- Knowledge representation
- Version control
"""

from .human import HUMAN
from .config import Config
from .pattern_recognition import PatternRecognitionSystem
from .resource_monitor import ResourceMonitor
from .knowledge_representation import KnowledgeGraph
from .version_control import VersionControl

__all__ = [
    'HUMAN',
    'Config',
    'PatternRecognitionSystem',
    'ResourceMonitor',
    'KnowledgeGraph',
    'VersionControl'
] 