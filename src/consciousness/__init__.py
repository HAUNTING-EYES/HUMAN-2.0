"""
HUMAN 2.0 - Autonomous Consciousness

The consciousness layer that enables true autonomy:
- Goal Engine: Autonomous goal generation
- Self-Monitor: Continuous self-assessment and self-awareness
- Autonomous Consciousness Loop: 24/7 autonomous operation

This system operates without human commands, continuously:
- Assessing its own state
- Generating goals based on needs
- Creating hierarchical plans
- Executing improvements
- Learning from outcomes
- Adapting strategies
"""

from .goal_engine import GoalEngine, Goal, GoalType
from .self_monitor import SelfMonitor, SelfAssessment
from .autonomous_consciousness import AutonomousConsciousness

__all__ = [
    'GoalEngine',
    'Goal',
    'GoalType',
    'SelfMonitor',
    'SelfAssessment',
    'AutonomousConsciousness'
]
