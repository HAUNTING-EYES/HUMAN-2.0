"""
HUMAN 2.0 - Multi-Agent System

8-agent autonomous swarm with event-driven coordination:
- Analyzer: Code analysis and quality assessment
- Improver: Code improvement generation (Claude 3.5)
- Tester: Auto-test generation and validation
- Learner: External knowledge acquisition (GitHub, web)
- Meta-Learning: Strategy optimization and pattern discovery
- Planner: Hierarchical planning (cycle, strategic, vision)
- Coordinator: Agent orchestration and task assignment
- Monitor: System health and resource monitoring
"""

from .base_agent import BaseAgent, AgentStatus, AgentMetrics
from .analyzer_agent import AnalyzerAgent
from .improver_agent import ImproverAgent
from .tester_agent import TesterAgent
from .learner_agent import LearnerAgent
from .meta_learning_agent import MetaLearningAgent
from .planner_agent import PlannerAgent
from .coordinator_agent import CoordinatorAgent
from .monitor_agent import MonitorAgent

__all__ = [
    'BaseAgent',
    'AgentStatus',
    'AgentMetrics',
    'AnalyzerAgent',
    'ImproverAgent',
    'TesterAgent',
    'LearnerAgent',
    'MetaLearningAgent',
    'PlannerAgent',
    'CoordinatorAgent',
    'MonitorAgent'
]
