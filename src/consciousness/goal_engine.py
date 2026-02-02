#!/usr/bin/env python3
"""
HUMAN 2.0 - Goal Engine
Autonomous goal generation based on system state.

Generates goals without human commands based on:
- Current system state
- Knowledge gaps
- Performance metrics
- Strategic plans
- Curiosity
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import sys
sys.path.append(str(Path(__file__).parent.parent))


class GoalType(Enum):
    """Types of goals"""
    COVERAGE = "coverage"  # Increase test coverage
    LEARNING = "learning"  # Learn new knowledge
    REFACTORING = "refactoring"  # Improve code quality
    PERFORMANCE = "performance"  # Optimize performance
    STRATEGIC = "strategic"  # Long-term strategic goal
    CURIOSITY = "curiosity"  # Curiosity-driven exploration


@dataclass
class Goal:
    """Autonomous goal"""
    goal_id: str
    goal_type: GoalType
    description: str
    priority: float  # 0-1
    target_metric: Optional[str] = None
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    status: str = "active"  # active, achieved, failed, abandoned
    created_at: datetime = None
    achieved_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if isinstance(self.goal_type, str):
            self.goal_type = GoalType(self.goal_type)

    @property
    def progress(self) -> float:
        """Calculate progress towards goal (0-1)"""
        if self.target_value is None or self.current_value is None:
            return 0.0

        if self.target_value == 0:
            return 1.0 if self.current_value == 0 else 0.0

        # For metrics where higher is better (e.g., coverage)
        progress = min(1.0, self.current_value / self.target_value)
        return max(0.0, progress)


class GoalEngine:
    """
    Autonomous goal generation engine.

    Generates goals based on system state without human input.
    """

    def __init__(self):
        """Initialize goal engine"""
        self.logger = logging.getLogger(__name__)

        # Active goals
        self.active_goals: List[Goal] = []
        self.achieved_goals: List[Goal] = []
        self.failed_goals: List[Goal] = []

        # Configuration
        self.config = {
            'max_active_goals': 5,
            'min_goal_priority': 0.3,
            'coverage_target': 0.75,
            'complexity_target': 8.0,
            'curiosity_enabled': True
        }

        self.logger.info("GoalEngine initialized")

    def generate_goals(self, assessment: Dict[str, Any]) -> List[Goal]:
        """
        Generate goals based on current assessment.

        Args:
            assessment: Self-assessment from SelfMonitor

        Returns:
            List of new goals
        """
        self.logger.info("Generating autonomous goals")

        goals = []

        # Goal 1: Coverage goals
        current_coverage = assessment.get('test_coverage', 0.0)
        if current_coverage < self.config['coverage_target']:
            goal = Goal(
                goal_id=f"coverage_{datetime.now().timestamp()}",
                goal_type=GoalType.COVERAGE,
                description=f"Increase test coverage to {self.config['coverage_target']:.0%}",
                priority=0.9,
                target_metric="test_coverage",
                target_value=self.config['coverage_target'],
                current_value=current_coverage
            )
            goals.append(goal)

        # Goal 2: Complexity reduction
        avg_complexity = assessment.get('avg_complexity', 0.0)
        if avg_complexity > self.config['complexity_target']:
            goal = Goal(
                goal_id=f"complexity_{datetime.now().timestamp()}",
                goal_type=GoalType.REFACTORING,
                description=f"Reduce average complexity to <{self.config['complexity_target']}",
                priority=0.8,
                target_metric="avg_complexity",
                target_value=self.config['complexity_target'],
                current_value=avg_complexity
            )
            goals.append(goal)

        # Goal 3: Learning (curiosity-driven)
        if self.config['curiosity_enabled']:
            knowledge_gaps = assessment.get('knowledge_gaps', [])
            for gap in knowledge_gaps[:2]:  # Top 2 gaps
                goal = Goal(
                    goal_id=f"learning_{gap['topic']}_{datetime.now().timestamp()}",
                    goal_type=GoalType.LEARNING,
                    description=f"Learn about {gap['topic']}",
                    priority=0.6,
                    target_metric="knowledge_acquired",
                    target_value=1.0
                )
                goals.append(goal)

        # Goal 4: Bug fixes
        bug_count = assessment.get('bug_count', 0)
        if bug_count > 0:
            goal = Goal(
                goal_id=f"bugs_{datetime.now().timestamp()}",
                goal_type=GoalType.REFACTORING,
                description=f"Fix {bug_count} known bugs",
                priority=1.0,  # High priority
                target_metric="bug_count",
                target_value=0,
                current_value=bug_count
            )
            goals.append(goal)

        # Goal 5: Strategic goals from plan
        strategic_goals = assessment.get('strategic_goals', [])
        for sg in strategic_goals[:2]:
            goal = Goal(
                goal_id=f"strategic_{datetime.now().timestamp()}",
                goal_type=GoalType.STRATEGIC,
                description=sg.get('description', 'Strategic goal'),
                priority=sg.get('priority', 0.7)
            )
            goals.append(goal)

        # Filter by priority
        goals = [g for g in goals if g.priority >= self.config['min_goal_priority']]

        # Sort by priority
        goals.sort(key=lambda g: g.priority, reverse=True)

        self.logger.info(f"Generated {len(goals)} new goals")

        return goals

    def add_goals(self, goals: List[Goal]):
        """Add goals to active list"""
        for goal in goals:
            if len(self.active_goals) >= self.config['max_active_goals']:
                break
            self.active_goals.append(goal)
            self.logger.info(f"Added goal: {goal.description} (priority: {goal.priority:.2f})")

    def get_priority_goals(self, top_n: int = 3) -> List[Goal]:
        """
        Get top N priority goals.

        Args:
            top_n: Number of goals to return

        Returns:
            Top priority goals
        """
        # Sort by priority
        sorted_goals = sorted(self.active_goals, key=lambda g: g.priority, reverse=True)
        return sorted_goals[:top_n]

    def update_goal_progress(self, goal_id: str, current_value: float):
        """
        Update progress on a goal.

        Args:
            goal_id: Goal ID
            current_value: Current metric value
        """
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.current_value = current_value

                # Check if achieved
                if goal.progress >= 1.0:
                    self._achieve_goal(goal)
                    break

    def _achieve_goal(self, goal: Goal):
        """Mark goal as achieved"""
        goal.status = "achieved"
        goal.achieved_at = datetime.now()
        self.active_goals.remove(goal)
        self.achieved_goals.append(goal)

        self.logger.info(f"Goal achieved: {goal.description}")

    def fail_goal(self, goal_id: str, reason: str = ""):
        """Mark goal as failed"""
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.status = "failed"
                self.active_goals.remove(goal)
                self.failed_goals.append(goal)
                self.logger.warning(f"Goal failed: {goal.description} - {reason}")
                break

    def abandon_goal(self, goal_id: str, reason: str = ""):
        """Abandon a goal"""
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.status = "abandoned"
                self.active_goals.remove(goal)
                self.logger.info(f"Goal abandoned: {goal.description} - {reason}")
                break

    def get_statistics(self) -> Dict[str, Any]:
        """Get goal statistics"""
        total_goals = len(self.active_goals) + len(self.achieved_goals) + len(self.failed_goals)

        return {
            'active_goals': len(self.active_goals),
            'achieved_goals': len(self.achieved_goals),
            'failed_goals': len(self.failed_goals),
            'total_goals': total_goals,
            'achievement_rate': len(self.achieved_goals) / total_goals if total_goals > 0 else 0.0,
            'avg_priority': sum(g.priority for g in self.active_goals) / len(self.active_goals) if self.active_goals else 0.0
        }


if __name__ == "__main__":
    # Test goal engine
    logging.basicConfig(level=logging.INFO)

    engine = GoalEngine()

    # Mock assessment
    assessment = {
        'test_coverage': 0.60,
        'avg_complexity': 12.0,
        'bug_count': 3,
        'knowledge_gaps': [
            {'topic': 'async patterns', 'importance': 0.8},
            {'topic': 'optimization techniques', 'importance': 0.7}
        ],
        'strategic_goals': [
            {'description': 'Achieve 80% coverage in 10 cycles', 'priority': 0.9}
        ]
    }

    # Generate goals
    goals = engine.generate_goals(assessment)
    print(f"\nGenerated {len(goals)} goals:")
    for goal in goals:
        print(f"  - {goal.description} (priority: {goal.priority:.2f}, progress: {goal.progress:.1%})")

    # Add to active goals
    engine.add_goals(goals)

    # Get priority goals
    priority_goals = engine.get_priority_goals(top_n=3)
    print(f"\nTop 3 Priority Goals:")
    for goal in priority_goals:
        print(f"  - {goal.description} ({goal.goal_type.value})")

    # Update progress
    if goals:
        engine.update_goal_progress(goals[0].goal_id, 0.75)

    # Get statistics
    stats = engine.get_statistics()
    print(f"\nGoal Statistics:")
    print(f"  Active: {stats['active_goals']}")
    print(f"  Achieved: {stats['achieved_goals']}")
    print(f"  Achievement Rate: {stats['achievement_rate']:.1%}")
