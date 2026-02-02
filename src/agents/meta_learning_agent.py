#!/usr/bin/env python3
"""
HUMAN 2.0 - Meta-Learning Agent
Learns how to learn better - optimizes strategies and approaches.

Responsibilities:
- Track improvement outcomes
- Identify successful strategies
- Optimize prompts and parameters
- Run A/B tests on approaches
- Update pattern library with successful patterns
- Learn from failures
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentStatus
from core.event_bus import EventBus, Event, EventTypes, EventPriority
from core.shared_resources import SharedResources, Pattern


@dataclass
class StrategyOutcome:
    """Outcome of a strategy"""
    strategy_id: str
    strategy_type: str  # "refactoring", "testing", "performance"
    parameters: Dict[str, Any]
    success: bool
    improvement_score: float  # 0-1
    context: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MetaLearningAgent(BaseAgent):
    """
    Agent responsible for meta-learning and strategy optimization.

    Subscribes to:
    - improvement_applied: Track outcomes
    - improvement_failed: Learn from failures
    - cycle_completed: Analyze cycle performance
    - tests_executed: Track test outcomes

    Publishes:
    - strategy_optimized: New strategy parameters
    - pattern_discovered: New successful pattern
    """

    def __init__(self, name: str, event_bus: EventBus, resources: SharedResources):
        """
        Initialize Meta-Learning Agent.

        Args:
            name: Agent name
            event_bus: Event bus for communication
            resources: Shared resources
        """
        super().__init__(name, event_bus)
        self.resources = resources

        # Track outcomes
        self.outcomes: List[StrategyOutcome] = []
        self.strategy_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'total_improvement': 0.0,
            'best_parameters': {}
        })

        # Configuration
        self.config = {
            'min_samples_for_optimization': 10,
            'pattern_discovery_threshold': 0.8,  # 80% success rate
            'learning_rate': 0.1
        }

        self.logger.info(f"MetaLearningAgent initialized")

    def register_event_handlers(self):
        """Register event handlers"""
        self.event_bus.subscribe(EventTypes.IMPROVEMENT_APPLIED, self.on_improvement_applied, self.name)
        self.event_bus.subscribe(EventTypes.IMPROVEMENT_FAILED, self.on_improvement_failed, self.name)
        self.event_bus.subscribe(EventTypes.CYCLE_COMPLETED, self.on_cycle_completed, self.name)
        self.event_bus.subscribe(EventTypes.TESTS_EXECUTED, self.on_tests_executed, self.name)
        self.logger.info(f"Subscribed to: improvement_applied, improvement_failed, cycle_completed, tests_executed")

    async def on_improvement_applied(self, event: Event):
        """Track successful improvement"""
        improvement = event.data.get('improvement', {})

        self.logger.info(f"Recording successful improvement: {improvement.get('improvement_type')}")

        outcome = StrategyOutcome(
            strategy_id=improvement.get('pattern_id', 'default'),
            strategy_type=improvement.get('improvement_type', 'unknown'),
            parameters={},
            success=True,
            improvement_score=improvement.get('estimated_impact', 0.5),
            context={
                'file_path': improvement.get('file_path'),
                'description': improvement.get('description')
            }
        )

        task = {
            'type': 'record_outcome',
            'outcome': asdict(outcome)
        }
        await self.execute_task(task)

    async def on_improvement_failed(self, event: Event):
        """Learn from failed improvement"""
        improvement = event.data.get('improvement', {})

        self.logger.info(f"Recording failed improvement: {improvement.get('improvement_type')}")

        outcome = StrategyOutcome(
            strategy_id=improvement.get('pattern_id', 'default'),
            strategy_type=improvement.get('improvement_type', 'unknown'),
            parameters={},
            success=False,
            improvement_score=0.0,
            context={
                'file_path': improvement.get('file_path'),
                'reason': event.data.get('reason', 'unknown')
            }
        )

        task = {
            'type': 'record_outcome',
            'outcome': asdict(outcome)
        }
        await self.execute_task(task)

    async def on_cycle_completed(self, event: Event):
        """Analyze cycle performance and optimize"""
        cycle_number = event.data.get('cycle_number')
        outcomes = event.data.get('outcomes', [])

        self.logger.info(f"Analyzing cycle {cycle_number} with {len(outcomes)} outcomes")

        task = {
            'type': 'optimize_strategies',
            'cycle_number': cycle_number,
            'outcomes': outcomes
        }
        await self.execute_task(task)

    async def on_tests_executed(self, event: Event):
        """Track test outcomes"""
        result = event.data.get('result', {})
        file_path = event.data.get('file_path')

        # Record test coverage in metrics
        coverage = result.get('coverage', 0.0)
        self.resources.record_metric(
            category='test_coverage',
            name=file_path,
            value=coverage
        )

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing: Meta-learning tasks.

        Args:
            task: Task with type (record_outcome/optimize_strategies)

        Returns:
            Processing result
        """
        task_type = task.get('type')

        if task_type == 'record_outcome':
            return await self._record_outcome(task.get('outcome'))
        elif task_type == 'optimize_strategies':
            return await self._optimize_strategies(task.get('outcomes', []))
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _record_outcome(self, outcome_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record improvement outcome.

        Args:
            outcome_data: Outcome data

        Returns:
            Record result
        """
        outcome = StrategyOutcome(**outcome_data)
        self.outcomes.append(outcome)

        # Update strategy stats
        stats = self.strategy_stats[outcome.strategy_type]
        stats['attempts'] += 1
        if outcome.success:
            stats['successes'] += 1
            stats['total_improvement'] += outcome.improvement_score

        # Update pattern library if pattern was used
        if outcome.strategy_id and outcome.strategy_id != 'default':
            self.resources.update_pattern_outcome(
                pattern_id=outcome.strategy_id,
                success=outcome.success,
                improvement=outcome.improvement_score
            )

        # Check if this creates a new pattern
        if outcome.success and outcome.improvement_score > self.config['pattern_discovery_threshold']:
            await self._discover_pattern(outcome)

        self.logger.debug(f"Recorded outcome: {outcome.strategy_type} (success={outcome.success})")

        return {
            'success': True,
            'outcome_recorded': True
        }

    async def _optimize_strategies(self, outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize strategies based on outcomes.

        Args:
            outcomes: List of outcomes from cycle

        Returns:
            Optimization result
        """
        self.logger.info(f"Optimizing strategies based on {len(outcomes)} outcomes")

        optimizations = []

        # Analyze each strategy type
        for strategy_type, stats in self.strategy_stats.items():
            if stats['attempts'] < self.config['min_samples_for_optimization']:
                continue

            success_rate = stats['successes'] / stats['attempts']
            avg_improvement = stats['total_improvement'] / stats['attempts'] if stats['attempts'] > 0 else 0

            self.logger.info(f"Strategy {strategy_type}: {success_rate:.1%} success, {avg_improvement:.2f} avg improvement")

            # If success rate is low, publish optimization suggestion
            if success_rate < 0.5:
                optimization = {
                    'strategy_type': strategy_type,
                    'current_success_rate': success_rate,
                    'suggestion': 'Reduce usage or modify parameters',
                    'parameters': {
                        'min_priority_for_auto_apply': 0.7  # Increase threshold
                    }
                }
                optimizations.append(optimization)

                # Publish strategy optimized event
                await self.publish_event(
                    EventTypes.STRATEGY_OPTIMIZED,
                    {
                        'strategy': strategy_type,
                        'optimization': optimization,
                        'parameters': optimization['parameters']
                    },
                    EventPriority.NORMAL
                )

        return {
            'success': True,
            'optimizations': optimizations,
            'strategies_analyzed': len(self.strategy_stats)
        }

    async def _discover_pattern(self, outcome: StrategyOutcome):
        """
        Discover new successful pattern.

        Args:
            outcome: Successful outcome
        """
        self.logger.info(f"Discovering pattern from outcome: {outcome.strategy_type}")

        # Create new pattern
        pattern = Pattern(
            pattern_id=f"discovered_{outcome.strategy_type}_{datetime.now().timestamp()}",
            name=f"Discovered {outcome.strategy_type.title()} Pattern",
            description=f"Pattern discovered from successful {outcome.strategy_type}",
            category=outcome.strategy_type,
            code_template="",  # Would extract from actual improvement
            success_count=1,
            failure_count=0,
            avg_improvement=outcome.improvement_score,
            tags=[outcome.strategy_type, 'discovered', 'high_impact']
        )

        # Add to pattern library
        self.resources.add_pattern(pattern)

        # Publish pattern discovered event
        await self.publish_event(
            EventTypes.PATTERN_DISCOVERED,
            {
                'pattern': asdict(pattern),
                'outcome': asdict(outcome)
            },
            EventPriority.HIGH
        )

        self.logger.info(f"New pattern discovered: {pattern.name}")

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics on all strategies"""
        stats = {}
        for strategy_type, data in self.strategy_stats.items():
            success_rate = data['successes'] / data['attempts'] if data['attempts'] > 0 else 0
            avg_improvement = data['total_improvement'] / data['attempts'] if data['attempts'] > 0 else 0

            stats[strategy_type] = {
                'attempts': data['attempts'],
                'successes': data['successes'],
                'success_rate': success_rate,
                'avg_improvement': avg_improvement
            }

        return stats

    def validate_output(self, output: Any) -> bool:
        """Validate meta-learning output"""
        if not isinstance(output, dict):
            return False

        if not output.get('success'):
            return True  # Errors are valid

        return True


if __name__ == "__main__":
    # Test meta-learning agent
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus, create_event
    from core.shared_resources import SharedResources

    async def test_meta_learning():
        """Test meta-learning agent"""
        bus = EventBus()
        resources = SharedResources()
        agent = MetaLearningAgent("meta-learner", bus, resources)

        # Simulate some outcomes
        for i in range(5):
            outcome = {
                'strategy_id': 'default',
                'strategy_type': 'refactoring',
                'parameters': {},
                'success': i % 2 == 0,  # 60% success
                'improvement_score': 0.7 if i % 2 == 0 else 0.0,
                'context': {'file': f'file_{i}.py'}
            }

            task = {
                'type': 'record_outcome',
                'outcome': outcome
            }
            await agent.execute_task(task)

        # Get stats
        stats = agent.get_strategy_stats()
        print(f"\nStrategy Stats:")
        for strategy_type, data in stats.items():
            print(f"  {strategy_type}:")
            print(f"    Attempts: {data['attempts']}")
            print(f"    Success Rate: {data['success_rate']:.1%}")
            print(f"    Avg Improvement: {data['avg_improvement']:.2f}")

        # Test optimization
        task = {
            'type': 'optimize_strategies',
            'outcomes': []
        }
        result = await agent.execute_task(task)
        print(f"\nOptimization Result:")
        print(f"  Success: {result.get('success')}")
        print(f"  Optimizations: {len(result.get('optimizations', []))}")

        # Get agent status
        status = agent.get_status()
        print(f"\nAgent Status:")
        print(f"  Tasks Processed: {status['metrics']['tasks_processed']}")

    asyncio.run(test_meta_learning())
