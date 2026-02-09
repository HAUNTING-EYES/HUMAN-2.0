#!/usr/bin/env python3
"""
HUMAN 2.0 - Event Bus System
Event-driven coordination for multi-agent architecture.
"""

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
from enum import Enum


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """
    Base event class for agent communication.

    Events are the primary communication mechanism between agents.
    They enable loose coupling and parallel execution.
    """
    type: str
    data: Dict[str, Any]
    source_agent: str
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S_%f'))

    def __repr__(self) -> str:
        return f"Event(type='{self.type}', source='{self.source_agent}', priority={self.priority.name})"


class EventBus:
    """
    Central event bus for inter-agent communication.

    Features:
    - Pub/sub pattern for loose coupling
    - Async/parallel event handling
    - Event history for debugging
    - Priority-based event processing
    - Event filtering
    """

    def __init__(self, max_history: int = 10000):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        self.is_running = True
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int)
        }

    def subscribe(self, event_type: str, handler: Callable, agent_name: str = "Unknown"):
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to (e.g., "code_analyzed")
            handler: Async function to call when event is published
            agent_name: Name of subscribing agent (for logging)
        """
        self.subscribers[event_type].append(handler)
        self.logger.info(f"Agent '{agent_name}' subscribed to event '{event_type}'")

    def subscribe_all(self, handler: Callable, agent_name: str = "Unknown"):
        """
        Subscribe to ALL events (useful for Monitor agent).

        Args:
            handler: Async function to call for all events
            agent_name: Name of subscribing agent
        """
        self.subscribers['*'].append(handler)
        self.logger.info(f"Agent '{agent_name}' subscribed to ALL events")

    async def publish(self, event: Event):
        """
        Publish an event to all subscribers (parallel execution).

        Args:
            event: Event to publish
        """
        if not self.is_running:
            self.logger.warning(f"Event bus stopped, ignoring event: {event.type}")
            return

        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)  # Remove oldest

        # Update stats
        self.stats['total_events'] += 1
        self.stats['events_by_type'][event.type] += 1

        self.logger.debug(f"Publishing event: {event}")

        # Get handlers for this event type + wildcard handlers
        handlers = self.subscribers.get(event.type, []) + self.subscribers.get('*', [])

        if not handlers:
            self.logger.debug(f"No subscribers for event: {event.type}")
            return

        # Call all handlers in parallel
        try:
            await asyncio.gather(*[self._safe_handle(handler, event) for handler in handlers])
        except Exception as e:
            self.logger.error(f"Error publishing event {event.type}: {e}")

    async def _safe_handle(self, handler: Callable, event: Event):
        """
        Safely execute event handler with error handling.

        Args:
            handler: Event handler function
            event: Event to handle
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            self.logger.error(f"Error in event handler for {event.type}: {e}")

    def get_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """
        Get event history, optionally filtered by type.

        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events to return

        Returns:
            List of events (most recent first)
        """
        if event_type:
            filtered = [e for e in self.event_history if e.type == event_type]
        else:
            filtered = self.event_history

        return filtered[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            'total_events': self.stats['total_events'],
            'events_by_type': dict(self.stats['events_by_type']),
            'total_subscribers': sum(len(handlers) for handlers in self.subscribers.values()),
            'history_size': len(self.event_history)
        }

    def clear_history(self):
        """Clear event history (useful for testing)."""
        self.event_history.clear()
        self.logger.info("Event history cleared")

    def stop(self):
        """Stop the event bus (no new events will be processed)."""
        self.is_running = False
        self.logger.warning("Event bus stopped")

    def start(self):
        """Start the event bus."""
        self.is_running = True
        self.logger.info("Event bus started")


# Pre-defined event types for the multi-agent system
class EventTypes:
    """Standard event types used by agents"""

    # Analysis events
    CODE_ANALYZED = "code_analyzed"
    ANALYSIS_REQUESTED = "analysis_requested"

    # Improvement events
    IMPROVEMENT_PROPOSED = "improvement_proposed"
    IMPROVEMENT_APPLIED = "improvement_applied"
    IMPROVEMENT_FAILED = "improvement_failed"

    # Test events
    TESTS_GENERATED = "tests_generated"
    TESTS_EXECUTED = "tests_executed"
    TEST_COVERAGE_UPDATED = "test_coverage_updated"

    # Learning events
    KNOWLEDGE_ACQUIRED = "knowledge_acquired"
    KNOWLEDGE_GAP_IDENTIFIED = "knowledge_gap_identified"
    CURIOSITY_TRIGGERED = "curiosity_triggered"

    # Meta-learning events
    STRATEGY_OPTIMIZED = "strategy_optimized"
    PATTERN_DISCOVERED = "pattern_discovered"

    # Planning events
    CYCLE_PLAN_CREATED = "cycle_plan_created"
    STRATEGIC_PLAN_CREATED = "strategic_plan_created"
    VISION_PLAN_CREATED = "vision_plan_created"

    # Coordination events
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"

    # Goal events
    GOAL_CREATED = "goal_created"
    GOAL_ACHIEVED = "goal_achieved"
    GOAL_FAILED = "goal_failed"

    # Cycle events
    CYCLE_STARTED = "cycle_started"
    CYCLE_COMPLETED = "cycle_completed"

    # System events
    HEALTH_STATUS = "health_status"
    ANOMALY_DETECTED = "anomaly_detected"
    ERROR_OCCURRED = "error_occurred"
    RESOURCE_WARNING = "resource_warning"


def create_event(event_type: str, data: Dict[str, Any], source_agent: str,
                priority: EventPriority = EventPriority.NORMAL) -> Event:
    """
    Helper function to create an event.

    Args:
        event_type: Type of event (use EventTypes constants)
        data: Event data
        source_agent: Name of agent creating the event
        priority: Event priority

    Returns:
        Created event
    """
    return Event(
        type=event_type,
        data=data,
        source_agent=source_agent,
        priority=priority
    )


# Singleton event bus instance
_event_bus = None

def get_event_bus() -> EventBus:
    """Get the singleton event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


if __name__ == "__main__":
    # Test the event bus
    logging.basicConfig(level=logging.INFO)

    async def test_event_bus():
        """Test event bus functionality"""
        bus = EventBus()

        # Test subscriber
        async def handler1(event: Event):
            print(f"Handler 1 received: {event.type} from {event.source_agent}")

        async def handler2(event: Event):
            print(f"Handler 2 received: {event.type} with data: {event.data}")

        # Subscribe
        bus.subscribe(EventTypes.CODE_ANALYZED, handler1, "TestAgent1")
        bus.subscribe(EventTypes.CODE_ANALYZED, handler2, "TestAgent2")

        # Publish
        event = create_event(
            EventTypes.CODE_ANALYZED,
            {'file': 'test.py', 'complexity': 5},
            'AnalyzerAgent'
        )
        await bus.publish(event)

        # Check stats
        stats = bus.get_stats()
        print(f"\nStats: {stats}")

        # Check history
        history = bus.get_history(EventTypes.CODE_ANALYZED)
        print(f"\nHistory: {len(history)} events")

    asyncio.run(test_event_bus())
