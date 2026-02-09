"""
Thought Trace System
Records agent reasoning processes for transparency and debugging.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from pathlib import Path


class ThoughtType(Enum):
    """Types of thoughts"""
    OBSERVATION = "observation"  # What the agent observes
    REASONING = "reasoning"  # How the agent reasons
    DECISION = "decision"  # What the agent decides
    ACTION = "action"  # What the agent does
    REFLECTION = "reflection"  # What the agent learns


@dataclass
class Thought:
    """A single thought in the reasoning process"""
    timestamp: datetime
    agent_name: str
    thought_type: ThoughtType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'agent_name': self.agent_name,
            'thought_type': self.thought_type.value,
            'content': self.content,
            'metadata': self.metadata
        }


@dataclass
class ThoughtTrace:
    """A complete trace of thoughts for a task"""
    task_id: str
    agent_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    thoughts: List[Thought] = field(default_factory=list)
    outcome: Optional[str] = None
    success: bool = False

    def add_thought(self, thought_type: ThoughtType, content: str, metadata: Dict[str, Any] = None):
        """Add a thought to the trace"""
        thought = Thought(
            timestamp=datetime.now(),
            agent_name=self.agent_name,
            thought_type=thought_type,
            content=content,
            metadata=metadata or {}
        )
        self.thoughts.append(thought)

    def complete(self, success: bool, outcome: str = None):
        """Mark the trace as complete"""
        self.completed_at = datetime.now()
        self.success = success
        self.outcome = outcome

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'agent_name': self.agent_name,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'thoughts': [t.to_dict() for t in self.thoughts],
            'outcome': self.outcome,
            'success': self.success,
            'duration_seconds': (self.completed_at - self.started_at).total_seconds() if self.completed_at else None
        }


class ThoughtTraceManager:
    """Manages thought traces for all agents"""

    def __init__(self, trace_dir: str = "data/thought_traces"):
        self.logger = logging.getLogger(__name__)
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.active_traces: Dict[str, ThoughtTrace] = {}  # task_id -> trace
        self.completed_traces: List[ThoughtTrace] = []

    def start_trace(self, task_id: str, agent_name: str) -> ThoughtTrace:
        """Start a new thought trace"""
        trace = ThoughtTrace(
            task_id=task_id,
            agent_name=agent_name,
            started_at=datetime.now()
        )
        self.active_traces[task_id] = trace
        self.logger.info(f"Started thought trace for task {task_id} by {agent_name}")
        return trace

    def get_trace(self, task_id: str) -> Optional[ThoughtTrace]:
        """Get an active trace"""
        return self.active_traces.get(task_id)

    def complete_trace(self, task_id: str, success: bool, outcome: str = None):
        """Complete a thought trace"""
        trace = self.active_traces.pop(task_id, None)
        if trace:
            trace.complete(success, outcome)
            self.completed_traces.append(trace)
            self._save_trace(trace)
            self.logger.info(f"Completed thought trace for task {task_id}: {outcome}")

    def _save_trace(self, trace: ThoughtTrace):
        """Save a trace to disk"""
        try:
            filename = f"{trace.task_id}_{trace.agent_name}_{trace.started_at.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.trace_dir / filename
            with open(filepath, 'w') as f:
                json.dump(trace.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save thought trace: {e}")

    def get_recent_traces(self, agent_name: str = None, limit: int = 10) -> List[ThoughtTrace]:
        """Get recent completed traces"""
        traces = self.completed_traces
        if agent_name:
            traces = [t for t in traces if t.agent_name == agent_name]
        return sorted(traces, key=lambda t: t.started_at, reverse=True)[:limit]

    def get_trace_statistics(self, agent_name: str = None) -> Dict[str, Any]:
        """Get statistics about thought traces"""
        traces = self.completed_traces
        if agent_name:
            traces = [t for t in traces if t.agent_name == agent_name]

        if not traces:
            return {
                'total_traces': 0,
                'success_rate': 0.0,
                'avg_thoughts_per_trace': 0.0,
                'avg_duration_seconds': 0.0
            }

        successful = [t for t in traces if t.success]
        avg_thoughts = sum(len(t.thoughts) for t in traces) / len(traces)
        completed = [t for t in traces if t.completed_at]
        avg_duration = sum((t.completed_at - t.started_at).total_seconds() for t in completed) / len(completed) if completed else 0

        return {
            'total_traces': len(traces),
            'success_rate': len(successful) / len(traces),
            'avg_thoughts_per_trace': avg_thoughts,
            'avg_duration_seconds': avg_duration,
            'thought_type_distribution': self._get_thought_type_distribution(traces)
        }

    def _get_thought_type_distribution(self, traces: List[ThoughtTrace]) -> Dict[str, int]:
        """Get distribution of thought types"""
        distribution = {tt.value: 0 for tt in ThoughtType}
        for trace in traces:
            for thought in trace.thoughts:
                distribution[thought.thought_type.value] += 1
        return distribution


class ThoughtTraceMixin:
    """Mixin to add thought tracing to agents"""

    def _init_thought_trace(self, trace_manager: ThoughtTraceManager):
        """Initialize thought tracing (call in agent __init__)"""
        self.trace_manager = trace_manager
        self.current_trace: Optional[ThoughtTrace] = None

    def _start_trace(self, task_id: str):
        """Start a thought trace for a task"""
        if hasattr(self, 'trace_manager') and self.trace_manager:
            self.current_trace = self.trace_manager.start_trace(task_id, self.name)

    def _observe(self, observation: str, metadata: Dict[str, Any] = None):
        """Record an observation"""
        if self.current_trace:
            self.current_trace.add_thought(ThoughtType.OBSERVATION, observation, metadata)

    def _reason(self, reasoning: str, metadata: Dict[str, Any] = None):
        """Record reasoning"""
        if self.current_trace:
            self.current_trace.add_thought(ThoughtType.REASONING, reasoning, metadata)

    def _decide(self, decision: str, metadata: Dict[str, Any] = None):
        """Record a decision"""
        if self.current_trace:
            self.current_trace.add_thought(ThoughtType.DECISION, decision, metadata)

    def _act(self, action: str, metadata: Dict[str, Any] = None):
        """Record an action"""
        if self.current_trace:
            self.current_trace.add_thought(ThoughtType.ACTION, action, metadata)

    def _reflect(self, reflection: str, metadata: Dict[str, Any] = None):
        """Record a reflection"""
        if self.current_trace:
            self.current_trace.add_thought(ThoughtType.REFLECTION, reflection, metadata)

    def _complete_trace(self, success: bool, outcome: str = None):
        """Complete the current trace"""
        if self.current_trace and hasattr(self, 'trace_manager'):
            self.trace_manager.complete_trace(self.current_trace.task_id, success, outcome)
            self.current_trace = None
