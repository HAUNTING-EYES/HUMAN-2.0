#!/usr/bin/env python3
"""
HUMAN 2.0 - Planner Agent
Creates hierarchical plans: cycle, strategic (10), and vision (20).

Responsibilities:
- Create cycle plans (what, how, aim)
- Create 10-cycle strategic plans
- Create 20-cycle vision plans
- Use MCTS for decision-making (future)
- Assign tasks to agents
- Define success criteria
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentStatus
from core.event_bus import EventBus, Event, EventTypes, EventPriority
from core.shared_resources import SharedResources


@dataclass
class Task:
    """A task in a plan"""
    task_id: str
    task_type: str  # "analyze", "improve", "test", "learn"
    target: str  # file_path or topic
    assigned_agent: str
    priority: float  # 0-1
    estimated_duration: int  # seconds
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Criterion:
    """Success criterion"""
    name: str
    target_value: Any
    weight: float  # 0-1
    current_value: Optional[Any] = None


@dataclass
class CyclePlan:
    """Plan for one improvement cycle"""
    cycle_number: int
    timestamp: datetime

    # WHAT - Goals
    goals: List[str]
    files_to_improve: List[str]
    learning_topics: List[str]

    # HOW - Execution
    tasks: List[Task]
    execution_order: List[str]  # task_ids in order
    strategies: Dict[str, str]  # file -> strategy name

    # AIM - Success criteria
    success_criteria: List[Criterion]
    expected_outcomes: Dict[str, Any]

    # Resources
    estimated_api_calls: int
    estimated_duration: int  # seconds

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class StrategicPlan:
    """Strategic plan for 10 cycles"""
    plan_id: str
    start_cycle: int
    end_cycle: int
    timestamp: datetime

    # Strategic goals
    strategic_goals: List[str]
    milestones: List[Dict[str, Any]]  # cycle_number, description, metrics

    # Themes
    focus_areas: List[str]  # e.g., "testing", "performance", "refactoring"
    skill_development: List[str]  # New skills to acquire

    # Metrics
    target_metrics: Dict[str, float]
    kpis: List[str]

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class VisionPlan:
    """Long-term vision for 20 cycles"""
    vision_id: str
    start_cycle: int
    end_cycle: int
    timestamp: datetime

    # Vision
    vision: str  # What we want to become
    mission: str  # Our purpose

    # Transformations
    transformations: List[str]
    capabilities_to_acquire: List[str]

    # Evolution roadmap
    phase_1_goals: List[str]  # Cycles 1-7
    phase_2_goals: List[str]  # Cycles 8-14
    phase_3_goals: List[str]  # Cycles 15-20

    # Target state
    target_metrics: Dict[str, float]
    breakthrough_goals: List[str]

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


class PlannerAgent(BaseAgent):
    """
    Agent responsible for creating all levels of plans.

    Subscribes to:
    - cycle_completed: Create next cycle plan
    - goal_created: Incorporate new goals

    Publishes:
    - cycle_plan_created: Cycle plan ready
    - strategic_plan_created: Strategic plan ready
    - vision_plan_created: Vision plan ready
    """

    def __init__(self, name: str, event_bus: EventBus, resources: SharedResources):
        """
        Initialize Planner Agent.

        Args:
            name: Agent name
            event_bus: Event bus for communication
            resources: Shared resources
        """
        super().__init__(name, event_bus)
        self.resources = resources

        # Track cycle number
        self.current_cycle = 0

        # Store plans
        self.cycle_plans: List[CyclePlan] = []
        self.strategic_plans: List[StrategicPlan] = []
        self.vision_plans: List[VisionPlan] = []

        # Configuration
        self.config = {
            'strategic_plan_interval': 10,  # cycles
            'vision_plan_interval': 20,  # cycles
            'max_files_per_cycle': 5,
            'max_learning_topics_per_cycle': 2
        }

        self.logger.info(f"PlannerAgent initialized")

    def register_event_handlers(self):
        """Register event handlers"""
        self.event_bus.subscribe(EventTypes.CYCLE_COMPLETED, self.on_cycle_completed, self.name)
        self.event_bus.subscribe(EventTypes.GOAL_CREATED, self.on_goal_created, self.name)
        self.logger.info(f"Subscribed to: cycle_completed, goal_created")

    async def on_cycle_completed(self, event: Event):
        """Create next cycle plan when cycle completes"""
        cycle_number = event.data.get('cycle_number')
        outcomes = event.data.get('outcomes', [])

        self.logger.info(f"Cycle {cycle_number} completed, planning next cycle")

        self.current_cycle = cycle_number + 1

        # Create cycle plan
        task = {
            'type': 'cycle_plan',
            'cycle_number': self.current_cycle,
            'previous_outcomes': outcomes
        }
        await self.execute_task(task)

        # Check if strategic plan needed
        if self.current_cycle % self.config['strategic_plan_interval'] == 0:
            task = {
                'type': 'strategic_plan',
                'start_cycle': self.current_cycle,
                'end_cycle': self.current_cycle + 9
            }
            await self.execute_task(task)

        # Check if vision plan needed
        if self.current_cycle % self.config['vision_plan_interval'] == 0:
            task = {
                'type': 'vision_plan',
                'start_cycle': self.current_cycle,
                'end_cycle': self.current_cycle + 19
            }
            await self.execute_task(task)

    async def on_goal_created(self, event: Event):
        """Incorporate new goal into planning"""
        goal = event.data.get('goal')
        self.logger.info(f"New goal created: {goal}")
        # Would update current plans to incorporate new goal

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing: Create plans.

        Args:
            task: Task with plan type

        Returns:
            Plan result
        """
        task_type = task.get('type')

        if task_type == 'cycle_plan':
            return await self._create_cycle_plan(
                task.get('cycle_number'),
                task.get('previous_outcomes', [])
            )
        elif task_type == 'strategic_plan':
            return await self._create_strategic_plan(
                task.get('start_cycle'),
                task.get('end_cycle')
            )
        elif task_type == 'vision_plan':
            return await self._create_vision_plan(
                task.get('start_cycle'),
                task.get('end_cycle')
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _create_cycle_plan(self, cycle_number: int, previous_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create plan for one cycle.

        Args:
            cycle_number: Cycle number
            previous_outcomes: Outcomes from previous cycle

        Returns:
            Cycle plan
        """
        self.logger.info(f"Creating cycle plan #{cycle_number}")

        # Get current state
        test_coverage = self.resources.get_latest_metric('test_coverage', 'overall') or 0.6

        # Determine goals
        goals = []
        if test_coverage < 0.75:
            goals.append(f"Increase test coverage to 75% (current: {test_coverage:.1%})")

        # Get files to improve (prioritize by analysis)
        files_to_improve = await self._select_files_for_improvement()

        # Select learning topics
        learning_topics = await self._select_learning_topics()

        # Create tasks
        tasks = []
        task_counter = 0

        # Analysis tasks
        for file_path in files_to_improve:
            tasks.append(Task(
                task_id=f"task_{cycle_number}_{task_counter}",
                task_type="analyze",
                target=file_path,
                assigned_agent="analyzer",
                priority=0.8,
                estimated_duration=30
            ))
            task_counter += 1

        # Improvement tasks (depend on analysis)
        for file_path in files_to_improve:
            analysis_task_id = f"task_{cycle_number}_{files_to_improve.index(file_path)}"
            tasks.append(Task(
                task_id=f"task_{cycle_number}_{task_counter}",
                task_type="improve",
                target=file_path,
                assigned_agent="improver",
                priority=0.7,
                estimated_duration=120,
                dependencies=[analysis_task_id]
            ))
            task_counter += 1

        # Learning tasks
        for topic in learning_topics:
            tasks.append(Task(
                task_id=f"task_{cycle_number}_{task_counter}",
                task_type="learn",
                target=topic,
                assigned_agent="learner",
                priority=0.5,
                estimated_duration=60
            ))
            task_counter += 1

        # Create execution order (topological sort)
        execution_order = self._create_execution_order(tasks)

        # Define success criteria
        success_criteria = [
            Criterion(
                name="test_coverage",
                target_value=0.75,
                weight=0.4,
                current_value=test_coverage
            ),
            Criterion(
                name="improvements_applied",
                target_value=len(files_to_improve),
                weight=0.3
            ),
            Criterion(
                name="knowledge_acquired",
                target_value=len(learning_topics),
                weight=0.3
            )
        ]

        # Create cycle plan
        cycle_plan = CyclePlan(
            cycle_number=cycle_number,
            timestamp=datetime.now(),
            goals=goals,
            files_to_improve=files_to_improve,
            learning_topics=learning_topics,
            tasks=tasks,
            execution_order=execution_order,
            strategies={file: "refactoring" for file in files_to_improve},
            success_criteria=success_criteria,
            expected_outcomes={
                'improvements': len(files_to_improve),
                'tests_generated': len([f for f in files_to_improve if not self._has_tests(f)]),
                'knowledge_nodes': len(learning_topics)
            },
            estimated_api_calls=len(files_to_improve) * 2,  # improve + test
            estimated_duration=sum(t.estimated_duration for t in tasks)
        )

        # Store plan
        self.cycle_plans.append(cycle_plan)

        # Publish event
        await self.publish_event(
            EventTypes.CYCLE_PLAN_CREATED,
            {
                'cycle_plan': asdict(cycle_plan)
            },
            EventPriority.HIGH
        )

        self.logger.info(f"Cycle plan created: {len(tasks)} tasks, {len(goals)} goals")

        return {
            'success': True,
            'cycle_plan': asdict(cycle_plan)
        }

    async def _create_strategic_plan(self, start_cycle: int, end_cycle: int) -> Dict[str, Any]:
        """Create strategic plan for 10 cycles"""
        self.logger.info(f"Creating strategic plan: cycles {start_cycle}-{end_cycle}")

        strategic_plan = StrategicPlan(
            plan_id=f"strategic_{start_cycle}_{end_cycle}",
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            timestamp=datetime.now(),
            strategic_goals=[
                "Achieve 80% test coverage",
                "Eliminate circular dependencies",
                "Master async programming patterns"
            ],
            milestones=[
                {
                    'cycle': start_cycle + 3,
                    'description': '70% test coverage achieved',
                    'metrics': {'test_coverage': 0.7}
                },
                {
                    'cycle': start_cycle + 6,
                    'description': 'Zero circular dependencies',
                    'metrics': {'circular_deps': 0}
                },
                {
                    'cycle': end_cycle,
                    'description': '80% test coverage achieved',
                    'metrics': {'test_coverage': 0.8}
                }
            ],
            focus_areas=["testing", "refactoring", "learning"],
            skill_development=["async patterns", "testing strategies"],
            target_metrics={
                'test_coverage': 0.8,
                'avg_complexity': 8.0,
                'code_smells': 10
            },
            kpis=['test_coverage', 'complexity', 'maintainability']
        )

        self.strategic_plans.append(strategic_plan)

        await self.publish_event(
            EventTypes.STRATEGIC_PLAN_CREATED,
            {
                'strategic_plan': asdict(strategic_plan)
            },
            EventPriority.NORMAL
        )

        return {
            'success': True,
            'strategic_plan': asdict(strategic_plan)
        }

    async def _create_vision_plan(self, start_cycle: int, end_cycle: int) -> Dict[str, Any]:
        """Create vision plan for 20 cycles"""
        self.logger.info(f"Creating vision plan: cycles {start_cycle}-{end_cycle}")

        vision_plan = VisionPlan(
            vision_id=f"vision_{start_cycle}_{end_cycle}",
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            timestamp=datetime.now(),
            vision="Become fully autonomous AGI with 99% self-sufficiency",
            mission="Continuous self-improvement and knowledge acquisition",
            transformations=[
                "From reactive to proactive",
                "From narrow to general capabilities",
                "From single-agent to swarm intelligence"
            ],
            capabilities_to_acquire=[
                "Multi-language support",
                "Real-time learning",
                "Novel pattern discovery"
            ],
            phase_1_goals=["80% coverage", "Master async"],
            phase_2_goals=["Novel patterns", "Knowledge graph with 10k+ nodes"],
            phase_3_goals=["99% autonomous", "Zero manual intervention"],
            target_metrics={
                'test_coverage': 0.95,
                'autonomy': 0.99,
                'knowledge_nodes': 10000
            },
            breakthrough_goals=[
                "Discover novel optimization patterns",
                "Self-generate improvement strategies"
            ]
        )

        self.vision_plans.append(vision_plan)

        await self.publish_event(
            EventTypes.VISION_PLAN_CREATED,
            {
                'vision_plan': asdict(vision_plan)
            },
            EventPriority.NORMAL
        )

        return {
            'success': True,
            'vision_plan': asdict(vision_plan)
        }

    async def _select_files_for_improvement(self) -> List[str]:
        """
        Select files to improve based on:
        1. Code quality metrics (complexity, maintainability)
        2. Test coverage gaps
        3. Recent changes (files being actively developed)
        4. Diverse exploration (not same files every time)
        """
        from pathlib import Path
        import random

        # Get all Python files in src/
        src_dir = Path('src')
        all_files = []

        for py_file in src_dir.rglob('*.py'):
            # Skip test files, __pycache__, etc.
            if '__pycache__' in str(py_file) or 'test_' in py_file.name:
                continue
            all_files.append(str(py_file))

        if not all_files:
            return []

        # Get metrics from shared resources if available
        scored_files = []

        for file_path in all_files:
            try:
                # Simple scoring: prioritize files that need work
                score = 0

                # Read file to check complexity
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())

                    # Prioritize larger files (more opportunity for improvement)
                    score += min(lines / 100, 10)

                    # Prioritize files with todos/fixmes
                    score += content.lower().count('todo') * 2
                    score += content.lower().count('fixme') * 3

                    # Deprioritize recently improved files (explore diverse codebase)
                    # This prevents getting stuck on same files

                scored_files.append((score, file_path))

            except Exception:
                continue

        # Sort by score and add randomness for exploration
        scored_files.sort(reverse=True)

        # Take top candidates but shuffle to avoid always picking same files
        max_files = self.config.get('max_files_per_cycle', 2)
        top_candidates = scored_files[:max_files * 3]  # 3x candidates

        # Shuffle and pick
        random.shuffle(top_candidates)
        selected = [f[1] for f in top_candidates[:max_files]]

        self.logger.info(f"Selected {len(selected)} files for improvement from {len(all_files)} total")
        return selected

    async def _select_learning_topics(self) -> List[str]:
        """
        Select topics to learn from:
        1. Curiosity engine questions
        2. Knowledge gaps identified in code
        3. Emerging patterns from GitHub
        4. Technology trends
        """
        topics = []

        # Get curiosity-driven topics if curiosity engine available
        if hasattr(self.resources, 'curiosity_engine'):
            try:
                questions = self.resources.curiosity_engine.generate_questions(count=3)
                topics.extend([q.content for q in questions])
            except Exception:
                pass

        # Fallback: diverse learning topics for continuous growth
        fallback_topics = [
            'Advanced Python async patterns and event loops',
            'Neural architecture search and AutoML',
            'Meta-learning and few-shot learning',
            'Graph neural networks for code analysis',
            'Reinforcement learning for code optimization',
            'Program synthesis and code generation',
            'Static analysis and symbolic execution',
            'Distributed systems patterns',
            'Quantum computing algorithms',
            'Attention mechanisms and transformers'
        ]

        if not topics:
            import random
            topics = random.sample(fallback_topics, min(2, len(fallback_topics)))

        max_topics = self.config.get('max_learning_topics_per_cycle', 2)
        self.logger.info(f"Selected learning topics: {topics[:max_topics]}")
        return topics[:max_topics]

    def _create_execution_order(self, tasks: List[Task]) -> List[str]:
        """Create execution order respecting dependencies"""
        # Simple topological sort
        order = []
        remaining = {t.task_id: t for t in tasks}

        while remaining:
            # Find tasks with no unfulfilled dependencies
            ready = [
                tid for tid, task in remaining.items()
                if all(dep in order for dep in task.dependencies)
            ]

            if not ready:
                # Circular dependency or error
                # Add remaining tasks anyway
                ready = list(remaining.keys())

            for task_id in ready:
                order.append(task_id)
                del remaining[task_id]

        return order

    def _has_tests(self, file_path: str) -> bool:
        """Check if file has tests"""
        test_file = Path('tests') / f'test_{Path(file_path).name}'
        return test_file.exists()

    def validate_output(self, output: Any) -> bool:
        """Validate plan output"""
        if not isinstance(output, dict):
            return False

        if not output.get('success'):
            return True  # Errors are valid

        # Check for plan in output
        has_plan = ('cycle_plan' in output or
                   'strategic_plan' in output or
                   'vision_plan' in output)

        return has_plan


if __name__ == "__main__":
    # Test planner agent
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus, create_event
    from core.shared_resources import SharedResources

    async def test_planner():
        """Test planner agent"""
        bus = EventBus()
        resources = SharedResources()
        agent = PlannerAgent("planner", bus, resources)

        # Test cycle plan
        task = {
            'type': 'cycle_plan',
            'cycle_number': 1,
            'previous_outcomes': []
        }

        result = await agent.execute_task(task)
        print(f"\nCycle Plan Result:")
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            cycle_plan = result.get('cycle_plan', {})
            print(f"  Cycle: {cycle_plan.get('cycle_number')}")
            print(f"  Goals: {len(cycle_plan.get('goals', []))}")
            print(f"  Tasks: {len(cycle_plan.get('tasks', []))}")
            print(f"  Files: {len(cycle_plan.get('files_to_improve', []))}")
            print(f"  Learning Topics: {len(cycle_plan.get('learning_topics', []))}")

        # Test strategic plan
        task = {
            'type': 'strategic_plan',
            'start_cycle': 1,
            'end_cycle': 10
        }

        result = await agent.execute_task(task)
        print(f"\nStrategic Plan Result:")
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            plan = result.get('strategic_plan', {})
            print(f"  Goals: {len(plan.get('strategic_goals', []))}")
            print(f"  Milestones: {len(plan.get('milestones', []))}")

        # Get status
        status = agent.get_status()
        print(f"\nAgent Status:")
        print(f"  Tasks Processed: {status['metrics']['tasks_processed']}")

    asyncio.run(test_planner())
