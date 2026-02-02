#!/usr/bin/env python3
"""
HUMAN 2.0 - Coordinator Agent
Orchestrates all agents and executes cycle plans.

Responsibilities:
- Execute cycle plans
- Assign tasks to agents
- Resolve conflicts between agents
- Ensure safety (Layer 2 validation)
- Track cycle progress
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentStatus
from core.event_bus import EventBus, Event, EventTypes, EventPriority
from core.shared_resources import SharedResources


@dataclass
class CycleProgress:
    """Progress tracking for a cycle"""
    cycle_number: int
    start_time: datetime
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    outcomes: List[Dict[str, Any]]
    end_time: Optional[datetime] = None

    @property
    def is_complete(self) -> bool:
        return self.tasks_completed + self.tasks_failed >= self.tasks_total

    @property
    def success_rate(self) -> float:
        if self.tasks_completed + self.tasks_failed == 0:
            return 0.0
        return self.tasks_completed / (self.tasks_completed + self.tasks_failed)


class CoordinatorAgent(BaseAgent):
    """
    Agent responsible for coordinating all other agents.

    Subscribes to:
    - cycle_plan_created: Execute the plan
    - All task completion events: Track progress

    Publishes:
    - task_assigned: Task assigned to agent
    - cycle_completed: Cycle finished
    - conflict_detected: Agent conflict
    - conflict_resolved: Conflict resolved
    """

    def __init__(self, name: str, event_bus: EventBus, resources: SharedResources,
                 agents: Dict[str, BaseAgent] = None):
        """
        Initialize Coordinator Agent.

        Args:
            name: Agent name
            event_bus: Event bus for communication
            resources: Shared resources
            agents: Dictionary of all agents (name -> agent)
        """
        super().__init__(name, event_bus)
        self.resources = resources
        self.agents = agents or {}

        # Track cycles
        self.current_cycle: Optional[CycleProgress] = None
        self.cycle_history: List[CycleProgress] = []

        # Configuration
        self.config = {
            'max_parallel_tasks': 3,
            'task_timeout': 300,  # seconds
            'enable_conflict_resolution': True
        }

        self.logger.info(f"CoordinatorAgent initialized with {len(self.agents)} agents")

    def register_event_handlers(self):
        """Register event handlers"""
        self.event_bus.subscribe(EventTypes.CYCLE_PLAN_CREATED, self.on_cycle_plan_created, self.name)
        # Subscribe to all completion events
        self.event_bus.subscribe('*', self.on_any_event, self.name)
        self.logger.info(f"Subscribed to: cycle_plan_created, all events")

    async def on_cycle_plan_created(self, event: Event):
        """Execute cycle plan"""
        cycle_plan = event.data.get('cycle_plan')
        self.logger.info(f"Executing cycle plan #{cycle_plan.get('cycle_number')}")

        task = {
            'type': 'execute_cycle',
            'cycle_plan': cycle_plan
        }
        await self.execute_task(task)

    async def on_any_event(self, event: Event):
        """Track all events for progress monitoring"""
        # Update cycle progress based on events
        if self.current_cycle and event.type in [
            EventTypes.IMPROVEMENT_APPLIED,
            EventTypes.IMPROVEMENT_FAILED,
            EventTypes.TESTS_EXECUTED,
            EventTypes.KNOWLEDGE_ACQUIRED
        ]:
            if event.type == EventTypes.IMPROVEMENT_FAILED:
                self.current_cycle.tasks_failed += 1
            else:
                self.current_cycle.tasks_completed += 1

            self.current_cycle.outcomes.append({
                'event_type': event.type,
                'data': event.data,
                'timestamp': event.timestamp.isoformat()
            })

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing: Coordinate agents.

        Args:
            task: Task with type

        Returns:
            Coordination result
        """
        task_type = task.get('type')

        if task_type == 'execute_cycle':
            return await self._execute_cycle(task.get('cycle_plan'))
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _execute_cycle(self, cycle_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete cycle plan.

        Args:
            cycle_plan: Cycle plan to execute

        Returns:
            Execution result
        """
        cycle_number = cycle_plan.get('cycle_number')
        tasks = cycle_plan.get('tasks', [])
        execution_order = cycle_plan.get('execution_order', [])

        self.logger.info(f"Starting cycle #{cycle_number} with {len(tasks)} tasks")

        # Initialize progress tracking
        self.current_cycle = CycleProgress(
            cycle_number=cycle_number,
            start_time=datetime.now(),
            tasks_total=len(tasks),
            tasks_completed=0,
            tasks_failed=0,
            outcomes=[]
        )

        # Publish cycle started
        await self.publish_event(
            EventTypes.CYCLE_STARTED,
            {
                'cycle_number': cycle_number,
                'files_to_analyze': cycle_plan.get('files_to_improve', []),
                'learning_topics': cycle_plan.get('learning_topics', [])
            },
            EventPriority.HIGH
        )

        # Execute tasks - group by type for parallel execution
        task_map = {t['task_id']: t for t in tasks}

        # Group tasks by type
        tasks_by_type = {}
        for task_id in execution_order:
            task = task_map.get(task_id)
            if not task:
                continue

            task_type = task.get('task_type', 'unknown')
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append(task)

        # Execute task groups in parallel (max 10 at a time)
        max_parallel = self.config.get('max_parallel_tasks', 10)

        for task_type, task_group in tasks_by_type.items():
            self.logger.info(f"Executing {len(task_group)} {task_type} tasks in parallel")

            # Execute in batches of max_parallel
            for i in range(0, len(task_group), max_parallel):
                batch = task_group[i:i + max_parallel]

                try:
                    # Execute all tasks in batch concurrently
                    await asyncio.gather(
                        *[self._execute_task(task) for task in batch],
                        return_exceptions=True
                    )
                except Exception as e:
                    self.logger.error(f"Batch execution failed: {e}")
                    self.current_cycle.tasks_failed += len(batch)

        # Complete cycle
        self.current_cycle.end_time = datetime.now()
        self.cycle_history.append(self.current_cycle)

        # Publish cycle completed
        await self.publish_event(
            EventTypes.CYCLE_COMPLETED,
            {
                'cycle_number': cycle_number,
                'outcomes': self.current_cycle.outcomes,
                'success_rate': self.current_cycle.success_rate,
                'duration': (self.current_cycle.end_time - self.current_cycle.start_time).total_seconds()
            },
            EventPriority.HIGH
        )

        self.logger.info(f"Cycle #{cycle_number} completed: {self.current_cycle.success_rate:.1%} success")

        current_cycle_result = asdict(self.current_cycle)
        self.current_cycle = None

        return {
            'success': True,
            'cycle_result': current_cycle_result
        }

    async def _execute_task(self, task: Dict[str, Any]):
        """
        Execute a single task by assigning to agent.

        Args:
            task: Task to execute
        """
        assigned_agent = task.get('assigned_agent')
        task_type = task.get('task_type')
        target = task.get('target')

        self.logger.info(f"Assigning {task_type} task to {assigned_agent}: {target}")

        # Publish task assigned
        await self.publish_event(
            EventTypes.TASK_ASSIGNED,
            {
                'task_id': task.get('task_id'),
                'agent': assigned_agent,
                'task_type': task_type,
                'target': target
            },
            EventPriority.NORMAL
        )

        # Note: Actual task execution happens via event subscriptions
        # Agents will pick up events and process them
        # This is just for tracking and coordination

        # Wait a bit for task to process (in real implementation, would track properly)
        await asyncio.sleep(0.5)

    def validate_improvement(self, improvement: Dict[str, Any]) -> bool:
        """
        Layer 2 validation: Coordinator validates cross-agent operations.

        Args:
            improvement: Improvement to validate

        Returns:
            True if valid
        """
        # Check that all tests pass
        # Check that dependencies are intact
        # Check that quality metrics improved
        # This is a safety layer above agent-level validation

        self.logger.debug(f"Validating improvement (Layer 2)")

        # Basic validation
        if not improvement.get('improved_code'):
            return False

        # Would run more sophisticated checks here
        return True

    def validate_output(self, output: Any) -> bool:
        """Validate coordination output"""
        if not isinstance(output, dict):
            return False

        return output.get('success') is not None


if __name__ == "__main__":
    # Test coordinator agent
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus
    from core.shared_resources import SharedResources

    async def test_coordinator():
        """Test coordinator agent"""
        bus = EventBus()
        resources = SharedResources()
        agent = CoordinatorAgent("coordinator", bus, resources)

        # Create mock cycle plan
        cycle_plan = {
            'cycle_number': 1,
            'tasks': [
                {
                    'task_id': 'task_1',
                    'task_type': 'analyze',
                    'target': 'src/test.py',
                    'assigned_agent': 'analyzer',
                    'priority': 0.8
                },
                {
                    'task_id': 'task_2',
                    'task_type': 'improve',
                    'target': 'src/test.py',
                    'assigned_agent': 'improver',
                    'priority': 0.7
                }
            ],
            'execution_order': ['task_1', 'task_2'],
            'files_to_improve': ['src/test.py'],
            'learning_topics': []
        }

        task = {
            'type': 'execute_cycle',
            'cycle_plan': cycle_plan
        }

        result = await agent.execute_task(task)
        print(f"\nCycle Execution Result:")
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            cycle_result = result.get('cycle_result', {})
            print(f"  Tasks Total: {cycle_result.get('tasks_total')}")
            print(f"  Tasks Completed: {cycle_result.get('tasks_completed')}")
            print(f"  Success Rate: {cycle_result.get('success_rate', 0):.1%}")

        # Get status
        status = agent.get_status()
        print(f"\nAgent Status:")
        print(f"  Tasks Processed: {status['metrics']['tasks_processed']}")

    asyncio.run(test_coordinator())
