#!/usr/bin/env python3
"""
HUMAN 2.0 - Self-Monitor
Continuous self-monitoring and self-awareness.

Monitors:
- Own code quality and health
- Performance metrics
- Learning efficiency
- Progress on goals and plans
- Resource usage
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class SelfAssessment:
    """Complete self-assessment of system state"""
    timestamp: datetime

    # Code Health
    test_coverage: float  # 0-1
    avg_complexity: float
    maintainability_index: float  # 0-100
    bug_count: int
    code_smells: int

    # Performance
    improvement_success_rate: float  # 0-1
    learning_efficiency: float  # 0-1
    avg_improvement_time: float  # seconds
    test_pass_rate: float  # 0-1

    # Progress
    progress_on_10_cycle_plan: float  # 0-1
    progress_on_20_cycle_plan: float  # 0-1
    goals_achieved: int
    goals_active: int

    # Resources
    api_quota_remaining: int
    cpu_usage: float  # 0-100
    memory_usage: float  # 0-100

    # Knowledge
    knowledge_nodes: int
    patterns_discovered: int
    external_knowledge_acquired: int

    # Flags
    needs_new_goals: bool
    needs_strategic_planning: bool
    needs_intervention: bool
    health_status: str  # "healthy", "degraded", "unhealthy"

    # Identified Issues
    knowledge_gaps: List[Dict[str, Any]]
    bottlenecks: List[str]
    opportunities: List[str]

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


class SelfMonitor:
    """
    Self-monitoring system for autonomous self-awareness.

    Continuously assesses own state and identifies needs.
    """

    def __init__(self, resources, agents: Dict[str, Any] = None):
        """
        Initialize self-monitor.

        Args:
            resources: Shared resources
            agents: Dictionary of all agents
        """
        self.logger = logging.getLogger(__name__)
        self.resources = resources
        self.agents = agents or {}

        # Track assessments
        self.assessment_history: List[SelfAssessment] = []

        # Configuration
        self.config = {
            'healthy_coverage_threshold': 0.75,
            'healthy_complexity_threshold': 10.0,
            'healthy_success_rate': 0.7,
            'assessment_retention': 100  # Keep last N assessments
        }

        self.logger.info("SelfMonitor initialized")

    def assess_current_state(self) -> SelfAssessment:
        """
        Comprehensive self-assessment.

        Returns:
            Complete assessment of current state
        """
        self.logger.info("Performing self-assessment")

        # Calculate code health metrics
        test_coverage = self._calculate_test_coverage()
        avg_complexity = self._calculate_avg_complexity()
        maintainability = self._calculate_maintainability()
        bug_count = self._count_bugs()
        code_smells = self._count_code_smells()

        # Calculate performance metrics
        success_rate = self._calculate_success_rate()
        learning_efficiency = self._calculate_learning_efficiency()
        avg_time = self._calculate_avg_improvement_time()
        test_pass_rate = self._calculate_test_pass_rate()

        # Calculate progress metrics
        progress_10 = self._measure_progress(cycles=10)
        progress_20 = self._measure_progress(cycles=20)
        goals_achieved, goals_active = self._count_goals()

        # Get resource usage
        api_quota, cpu, memory = self._get_resource_usage()

        # Get knowledge metrics
        knowledge_nodes = len(self.resources.knowledge_graph)
        patterns = len(self.resources.pattern_library)
        external = self.resources.code_embedder.external_knowledge_collection.count()

        # Identify gaps and opportunities
        knowledge_gaps = self._identify_knowledge_gaps()
        bottlenecks = self._identify_bottlenecks()
        opportunities = self._identify_opportunities()

        # Determine flags
        needs_new_goals = goals_active < 3
        needs_strategic_planning = progress_10 < 0.5  # Behind on strategic plan
        needs_intervention = (
            test_coverage < 0.5 or
            success_rate < 0.3 or
            bug_count > 10
        )

        # Determine overall health
        health_status = self._determine_health_status(
            test_coverage, avg_complexity, success_rate
        )

        # Create assessment
        assessment = SelfAssessment(
            timestamp=datetime.now(),
            test_coverage=test_coverage,
            avg_complexity=avg_complexity,
            maintainability_index=maintainability,
            bug_count=bug_count,
            code_smells=code_smells,
            improvement_success_rate=success_rate,
            learning_efficiency=learning_efficiency,
            avg_improvement_time=avg_time,
            test_pass_rate=test_pass_rate,
            progress_on_10_cycle_plan=progress_10,
            progress_on_20_cycle_plan=progress_20,
            goals_achieved=goals_achieved,
            goals_active=goals_active,
            api_quota_remaining=api_quota,
            cpu_usage=cpu,
            memory_usage=memory,
            knowledge_nodes=knowledge_nodes,
            patterns_discovered=patterns,
            external_knowledge_acquired=external,
            needs_new_goals=needs_new_goals,
            needs_strategic_planning=needs_strategic_planning,
            needs_intervention=needs_intervention,
            health_status=health_status,
            knowledge_gaps=knowledge_gaps,
            bottlenecks=bottlenecks,
            opportunities=opportunities
        )

        # Store assessment
        self.assessment_history.append(assessment)
        if len(self.assessment_history) > self.config['assessment_retention']:
            self.assessment_history.pop(0)

        self.logger.info(f"Self-assessment complete: {health_status} (coverage: {test_coverage:.1%}, success: {success_rate:.1%})")

        return assessment

    def _calculate_test_coverage(self) -> float:
        """Calculate overall test coverage"""
        # Get latest coverage metric
        coverage = self.resources.get_latest_metric('test_coverage', 'overall')
        return coverage if coverage is not None else 0.6

    def _calculate_avg_complexity(self) -> float:
        """Calculate average code complexity"""
        complexity = self.resources.get_latest_metric('complexity', 'avg')
        return complexity if complexity is not None else 10.0

    def _calculate_maintainability(self) -> float:
        """Calculate maintainability index"""
        maintainability = self.resources.get_latest_metric('maintainability', 'avg')
        return maintainability if maintainability is not None else 65.0

    def _count_bugs(self) -> int:
        """Count known bugs"""
        bugs = self.resources.get_latest_metric('bugs', 'count')
        return int(bugs) if bugs is not None else 0

    def _count_code_smells(self) -> int:
        """Count code smells"""
        smells = self.resources.get_latest_metric('code_smells', 'count')
        return int(smells) if smells is not None else 0

    def _calculate_success_rate(self) -> float:
        """Calculate improvement success rate"""
        # Get from meta-learner stats
        success_rate = self.resources.get_latest_metric('improvement', 'success_rate')
        return success_rate if success_rate is not None else 0.5

    def _calculate_learning_efficiency(self) -> float:
        """Calculate learning efficiency"""
        efficiency = self.resources.get_latest_metric('learning', 'efficiency')
        return efficiency if efficiency is not None else 0.6

    def _calculate_avg_improvement_time(self) -> float:
        """Calculate average time per improvement"""
        avg_time = self.resources.get_latest_metric('improvement', 'avg_time')
        return avg_time if avg_time is not None else 120.0

    def _calculate_test_pass_rate(self) -> float:
        """Calculate test pass rate"""
        pass_rate = self.resources.get_latest_metric('tests', 'pass_rate')
        return pass_rate if pass_rate is not None else 0.8

    def _measure_progress(self, cycles: int) -> float:
        """Measure progress on N-cycle plan"""
        # Would calculate based on actual plan progress
        # For now, return mock value
        return 0.65

    def _count_goals(self) -> tuple:
        """Count achieved and active goals"""
        # Would get from goal engine
        return (5, 3)  # (achieved, active)

    def _get_resource_usage(self) -> tuple:
        """Get resource usage metrics"""
        # Would get from monitor agent
        return (100000, 25.0, 40.0)  # (API quota, CPU%, Memory%)

    def _identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify knowledge gaps"""
        gaps = []

        # Check for missing expertise areas
        # In real implementation, would analyze based on failures and needs
        gaps.append({
            'topic': 'advanced async patterns',
            'importance': 0.8,
            'reason': 'Multiple async-related improvements failed'
        })

        return gaps

    def _identify_bottlenecks(self) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []

        # Check for performance issues
        if self._calculate_avg_improvement_time() > 180:
            bottlenecks.append("Slow improvement generation (>3min avg)")

        return bottlenecks

    def _identify_opportunities(self) -> List[str]:
        """Identify improvement opportunities"""
        opportunities = []

        # Check for quick wins
        if self._calculate_test_coverage() < 0.75:
            opportunities.append("Many untested files - easy coverage gains")

        if len(self.resources.pattern_library) < 10:
            opportunities.append("Few patterns discovered - potential for pattern mining")

        return opportunities

    def _determine_health_status(self, coverage: float, complexity: float, success_rate: float) -> str:
        """Determine overall health status"""
        issues = 0

        if coverage < self.config['healthy_coverage_threshold']:
            issues += 1
        if complexity > self.config['healthy_complexity_threshold']:
            issues += 1
        if success_rate < self.config['healthy_success_rate']:
            issues += 1

        if issues == 0:
            return "healthy"
        elif issues == 1:
            return "degraded"
        else:
            return "unhealthy"

    def update_understanding(self, outcomes: List[Dict[str, Any]]):
        """
        Update understanding based on cycle outcomes.

        Args:
            outcomes: Outcomes from completed cycle
        """
        self.logger.info(f"Updating understanding based on {len(outcomes)} outcomes")

        # Analyze outcomes
        successes = sum(1 for o in outcomes if o.get('success', False))
        failures = len(outcomes) - successes

        # Update metrics
        if outcomes:
            success_rate = successes / len(outcomes)
            self.resources.record_metric('improvement', 'success_rate', success_rate)

        self.logger.debug(f"Updated understanding: {successes}/{len(outcomes)} successful")

    def get_latest_assessment(self) -> Optional[SelfAssessment]:
        """Get the most recent assessment"""
        if self.assessment_history:
            return self.assessment_history[-1]
        return None

    def get_assessment_trend(self, metric: str, periods: int = 5) -> List[float]:
        """
        Get trend for a specific metric.

        Args:
            metric: Metric name (e.g., 'test_coverage')
            periods: Number of periods to return

        Returns:
            List of metric values over time
        """
        recent = self.assessment_history[-periods:]
        return [getattr(a, metric, 0) for a in recent]


if __name__ == "__main__":
    # Test self-monitor
    logging.basicConfig(level=logging.INFO)

    # Mock resources
    from core.shared_resources import SharedResources

    resources = SharedResources()
    monitor = SelfMonitor(resources)

    # Perform assessment
    assessment = monitor.assess_current_state()

    print(f"\nSelf-Assessment:")
    print(f"  Health: {assessment.health_status}")
    print(f"  Test Coverage: {assessment.test_coverage:.1%}")
    print(f"  Avg Complexity: {assessment.avg_complexity:.1f}")
    print(f"  Success Rate: {assessment.improvement_success_rate:.1%}")
    print(f"  Goals: {assessment.goals_active} active, {assessment.goals_achieved} achieved")
    print(f"  Knowledge Nodes: {assessment.knowledge_nodes}")
    print(f"  Needs New Goals: {assessment.needs_new_goals}")
    print(f"  Needs Intervention: {assessment.needs_intervention}")

    if assessment.knowledge_gaps:
        print(f"\nKnowledge Gaps:")
        for gap in assessment.knowledge_gaps:
            print(f"  - {gap['topic']} (importance: {gap['importance']:.1f})")

    if assessment.opportunities:
        print(f"\nOpportunities:")
        for opp in assessment.opportunities:
            print(f"  - {opp}")
