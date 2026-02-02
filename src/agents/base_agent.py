#!/usr/bin/env python3
"""
HUMAN 2.0 - Base Agent Class
Abstract base class for all agents in the multi-agent system.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.event_bus import EventBus, Event, create_event, EventTypes, EventPriority
from core.thought_trace import ThoughtTraceManager, ThoughtTraceMixin, ThoughtType


class AgentStatus(Enum):
    """Agent status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentMetrics:
    """Metrics for an agent"""
    tasks_processed: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    last_activity: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.tasks_processed == 0:
            return 0.0
        return self.tasks_succeeded / self.tasks_processed

    @property
    def avg_processing_time(self) -> float:
        """Calculate average processing time"""
        if self.tasks_processed == 0:
            return 0.0
        return self.total_processing_time / self.tasks_processed


class BaseAgent(ABC, ThoughtTraceMixin):
    """
    Abstract base class for all agents.

    All agents must:
    1. Have a unique name
    2. Subscribe to relevant events
    3. Process tasks asynchronously
    4. Publish events when work is done
    5. Validate their own outputs
    6. Track their own metrics
    7. Record thought traces for transparency
    """

    def __init__(self, name: str, event_bus: EventBus, resources=None, config: Dict[str, Any] = None):
        self.name = name
        self.event_bus = event_bus
        self.logger = logging.getLogger(f"agents.{name}")
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.config = config or {}

        # Initialize thought tracing
        if resources and hasattr(resources, 'thought_trace_manager'):
            self._init_thought_trace(resources.thought_trace_manager)
        else:
            self.trace_manager = None
            self.current_trace = None

        # Register event handlers
        self.register_event_handlers()

        self.logger.info(f"Agent '{self.name}' initialized")

    @abstractmethod
    def register_event_handlers(self):
        """
        Register which events this agent handles.

        Example:
            self.event_bus.subscribe('code_analyzed', self.on_code_analyzed, self.name)
        """
        pass

    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method - each agent implements its own logic.

        Args:
            task: Task to process

        Returns:
            Result dictionary
        """
        pass

    def validate_output(self, output: Any) -> bool:
        """
        Validate agent's own output (agent-level safety).

        Args:
            output: Output to validate

        Returns:
            True if valid, False otherwise
        """
        # Base implementation - override in subclasses
        return output is not None

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with error handling and metrics tracking.

        Args:
            task: Task to execute

        Returns:
            Result dictionary
        """
        start_time = datetime.now()
        self.status = AgentStatus.BUSY
        self.metrics.tasks_processed += 1

        # Start thought trace
        task_id = task.get('id', f"{self.name}_{start_time.timestamp()}")
        self._start_trace(task_id)

        try:
            task_type = task.get('type', 'unknown')
            self.logger.info(f"Processing task: {task_type}")

            # Record observation
            self._observe(f"Received task: {task_type}", {'task': task})

            # Process task
            result = await self.process(task)

            # Record action
            self._act(f"Completed task: {task_type}", {'result': result})

            # Validate output
            if not self.validate_output(result):
                self._decide("Output validation failed", {'result': result})
                raise ValueError("Output validation failed")

            # Record success
            self._reflect("Task completed successfully", {
                'success_rate': self.metrics.success_rate,
                'processing_time': (datetime.now() - start_time).total_seconds()
            })

            # Update metrics
            self.metrics.tasks_succeeded += 1
            self.status = AgentStatus.IDLE
            self.logger.info(f"Task completed successfully")

            # Complete trace
            self._complete_trace(success=True, outcome="Task completed successfully")

            return result

        except Exception as e:
            self.logger.error(f"Task failed: {e}")
            self.metrics.tasks_failed += 1
            self.status = AgentStatus.ERROR

            # Record failure
            self._reflect(f"Task failed: {str(e)}", {'error': str(e), 'task': task})

            # Complete trace
            self._complete_trace(success=False, outcome=f"Task failed: {str(e)}")

            # Publish error event
            await self.publish_event(
                EventTypes.ERROR_OCCURRED,
                {'agent': self.name, 'error': str(e), 'task': task},
                EventPriority.HIGH
            )

            return {'success': False, 'error': str(e)}

        finally:
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics.total_processing_time += processing_time
            self.metrics.last_activity = datetime.now()

    async def publish_event(self, event_type: str, data: Dict[str, Any],
                          priority: EventPriority = EventPriority.NORMAL):
        """
        Publish an event.

        Args:
            event_type: Type of event
            data: Event data
            priority: Event priority
        """
        event = create_event(event_type, data, self.name, priority)
        await self.event_bus.publish(event)

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status and metrics.

        Returns:
            Status dictionary
        """
        return {
            'name': self.name,
            'status': self.status.value,
            'metrics': {
                'tasks_processed': self.metrics.tasks_processed,
                'tasks_succeeded': self.metrics.tasks_succeeded,
                'tasks_failed': self.metrics.tasks_failed,
                'success_rate': self.metrics.success_rate,
                'avg_processing_time': self.metrics.avg_processing_time,
                'last_activity': self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            }
        }

    def configure(self, config: Dict[str, Any]):
        """
        Configure the agent.

        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        self.logger.info(f"Agent '{self.name}' configured with {len(config)} settings")

    def stop(self):
        """Stop the agent."""
        self.status = AgentStatus.STOPPED
        self.logger.info(f"Agent '{self.name}' stopped")

    def reset_metrics(self):
        """Reset metrics (useful for testing)."""
        self.metrics = AgentMetrics()
        self.logger.info(f"Agent '{self.name}' metrics reset")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status={self.status.value})"


if __name__ == "__main__":
    # Test base agent
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus

    class TestAgent(BaseAgent):
        """Simple test agent"""

        def register_event_handlers(self):
            self.event_bus.subscribe('test_event', self.on_test_event, self.name)

        async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate work
            return {'result': 'success', 'data': task.get('data', 'none')}

        async def on_test_event(self, event: Event):
            self.logger.info(f"Received test event: {event.data}")

    async def test_base_agent():
        bus = EventBus()
        agent = TestAgent("test-agent", bus)

        # Test task execution
        result = await agent.execute_task({'type': 'test', 'data': 'hello'})
        print(f"Result: {result}")

        # Test event
        await bus.publish(create_event('test_event', {'message': 'test'}, 'system'))

        # Get status
        status = agent.get_status()
        print(f"Status: {status}")

    asyncio.run(test_base_agent())
