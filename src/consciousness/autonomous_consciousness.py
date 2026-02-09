#!/usr/bin/env python3
"""
HUMAN 2.0 - Autonomous Consciousness
The main autonomous loop that operates 24/7 without human commands.

This is the "brain" that:
- Generates goals autonomously
- Creates plans
- Executes improvements
- Learns continuously
- Self-monitors
- Adapts strategies
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.event_bus import EventBus, EventTypes, create_event
# Lazy import to break circular dependency
_SharedResources = None
def get_shared_resources_class():
    global _SharedResources
    if _SharedResources is None:
        from core.shared_resources import SharedResources
        _SharedResources = SharedResources
    return _SharedResources
from consciousness.goal_engine import GoalEngine
from consciousness.self_monitor import SelfMonitor
from agents.planner_agent import PlannerAgent
from agents.coordinator_agent import CoordinatorAgent
from agents.monitor_agent import MonitorAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.improver_agent import ImproverAgent
from agents.tester_agent import TesterAgent
from agents.learner_agent import LearnerAgent
from agents.meta_learning_agent import MetaLearningAgent


class AutonomousConsciousness:
    """
    The autonomous consciousness loop.

    Runs continuously, generating goals, creating plans, and improving itself
    without any human intervention.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize autonomous consciousness.

        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Core systems
        self.event_bus = EventBus()
        self.resources = get_shared_resources_class()()
        self.goal_engine = GoalEngine()
        self.self_monitor = SelfMonitor(self.resources)

        # Initialize all agents
        self.agents = self._initialize_agents()

        # State
        self.is_paused = False
        self.is_running = False
        self.current_cycle = 0
        self.start_time: Optional[datetime] = None

        # Configuration
        self.config = {
            'consciousness_loop_interval': 300,  # 5 minutes between cycles
            'enable_autonomous_mode': True,
            'enable_goal_generation': True,
            'enable_strategic_planning': True,
            'max_cycles': None,  # None = run forever
            'sleep_between_cycles': 60  # seconds
        }

        # Load config if provided
        if config_path:
            self._load_config(config_path)

        self.logger.info("AutonomousConsciousness initialized")
        self.logger.info(f"Agents: {list(self.agents.keys())}")

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents"""
        self.logger.info("Initializing agent swarm...")

        agents = {}

        # Specialized agents
        agents['analyzer'] = AnalyzerAgent("analyzer", self.event_bus, self.resources)
        agents['improver'] = ImproverAgent("improver", self.event_bus, self.resources)
        agents['tester'] = TesterAgent("tester", self.event_bus, self.resources)
        agents['learner'] = LearnerAgent("learner", self.event_bus, self.resources)
        agents['meta_learning'] = MetaLearningAgent("meta_learning", self.event_bus, self.resources)

        # Coordination agents
        agents['planner'] = PlannerAgent("planner", self.event_bus, self.resources)
        agents['coordinator'] = CoordinatorAgent("coordinator", self.event_bus, self.resources, agents)
        agents['monitor'] = MonitorAgent("monitor", self.event_bus, self.resources, agents)

        self.logger.info(f"Initialized {len(agents)} agents")

        return agents

    async def consciousness_loop(self):
        """
        Main autonomous consciousness loop.

        This runs forever, continuously:
        1. Assessing current state
        2. Generating goals
        3. Creating plans
        4. Executing improvements
        5. Learning and adapting

        No human commands needed!
        """
        self.logger.info("=== STARTING AUTONOMOUS CONSCIOUSNESS LOOP ===")
        self.logger.info("System will now operate autonomously 24/7")

        self.is_running = True
        self.start_time = datetime.now()

        try:
            while self.is_running:
                # Check if paused
                if self.is_paused:
                    self.logger.info("Consciousness loop paused, sleeping...")
                    await asyncio.sleep(60)
                    continue

                # Check max cycles
                if self.config['max_cycles'] and self.current_cycle >= self.config['max_cycles']:
                    self.logger.info(f"Reached max cycles ({self.config['max_cycles']}), stopping")
                    break

                self.current_cycle += 1

                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"CYCLE #{self.current_cycle} - {datetime.now().isoformat()}")
                self.logger.info(f"{'='*60}")

                try:
                    # STEP 1: Self-Reflection (Self-Awareness)
                    self.logger.info("STEP 1: Performing self-assessment...")
                    assessment = self.self_monitor.assess_current_state()
                    self._log_assessment(assessment)

                    # STEP 2: Goal Generation (if needed)
                    if self.config['enable_goal_generation'] and assessment.needs_new_goals:
                        self.logger.info("STEP 2: Generating new goals...")
                        new_goals = self.goal_engine.generate_goals(assessment)
                        self.goal_engine.add_goals(new_goals)
                        self._log_new_goals(new_goals)
                    else:
                        self.logger.info("STEP 2: Sufficient active goals, skipping generation")

                    # STEP 3: Planning
                    self.logger.info("STEP 3: Creating cycle plan...")
                    cycle_plan = await self._create_cycle_plan(assessment)
                    self._log_cycle_plan(cycle_plan)

                    # STEP 4: Execution (Coordinator takes over)
                    self.logger.info("STEP 4: Executing cycle plan...")
                    outcomes = await self._execute_cycle(cycle_plan)
                    self._log_outcomes(outcomes)

                    # STEP 5: Learning (Update understanding)
                    self.logger.info("STEP 5: Updating understanding...")
                    self.self_monitor.update_understanding(outcomes)

                    # STEP 6: Strategic Planning (every 10 cycles)
                    if self.config['enable_strategic_planning'] and self.current_cycle % 10 == 0:
                        self.logger.info("STEP 6: Creating strategic plan (10 cycles)...")
                        await self._create_strategic_plan()

                    # STEP 7: Vision Planning (every 20 cycles)
                    if self.config['enable_strategic_planning'] and self.current_cycle % 20 == 0:
                        self.logger.info("STEP 7: Creating vision plan (20 cycles)...")
                        await self._create_vision_plan()

                    # STEP 8: Health Check
                    self.logger.info("STEP 8: Performing health check...")
                    await self._perform_health_check()

                    # STEP 9: Adaptive Sleep
                    sleep_duration = self._calculate_sleep_duration(assessment)
                    self.logger.info(f"STEP 9: Sleeping for {sleep_duration}s until next cycle...")
                    await asyncio.sleep(sleep_duration)

                except Exception as e:
                    self.logger.error(f"Error in consciousness loop cycle #{self.current_cycle}: {e}", exc_info=True)
                    # Continue to next cycle despite error
                    await asyncio.sleep(60)

        except KeyboardInterrupt:
            self.logger.info("Consciousness loop interrupted by user")
        finally:
            self.is_running = False
            self._log_shutdown()

    async def _create_cycle_plan(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Create cycle plan"""
        # Get priority goals
        priority_goals = self.goal_engine.get_priority_goals(top_n=3)

        # Planner will create the plan
        task = {
            'type': 'cycle_plan',
            'cycle_number': self.current_cycle,
            'previous_outcomes': []
        }

        result = await self.agents['planner'].execute_task(task)
        return result.get('cycle_plan', {})

    async def _execute_cycle(self, cycle_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute cycle plan via coordinator"""
        # Publish cycle plan created event
        await self.event_bus.publish(create_event(
            EventTypes.CYCLE_PLAN_CREATED,
            {'cycle_plan': cycle_plan},
            'consciousness'
        ))

        # Wait for cycle to complete (coordinator handles execution)
        # In real implementation, would await coordinator completion
        await asyncio.sleep(5)  # Simulate cycle execution

        # Return mock outcomes for now
        return [
            {'success': True, 'type': 'analysis'},
            {'success': True, 'type': 'improvement'}
        ]

    async def _create_strategic_plan(self):
        """Create strategic plan"""
        task = {
            'type': 'strategic_plan',
            'start_cycle': self.current_cycle,
            'end_cycle': self.current_cycle + 9
        }
        await self.agents['planner'].execute_task(task)

    async def _create_vision_plan(self):
        """Create vision plan"""
        task = {
            'type': 'vision_plan',
            'start_cycle': self.current_cycle,
            'end_cycle': self.current_cycle + 19
        }
        await self.agents['planner'].execute_task(task)

    async def _perform_health_check(self):
        """Perform system health check"""
        task = {'type': 'health_check'}
        await self.agents['monitor'].execute_task(task)

    def _calculate_sleep_duration(self, assessment: Dict[str, Any]) -> int:
        """Calculate adaptive sleep duration based on system state"""
        base_sleep = self.config['sleep_between_cycles']

        # Sleep longer if system is healthy and idle
        if assessment.health_status == "healthy" and assessment.goals_active == 0:
            return base_sleep * 2

        # Sleep less if urgent issues
        if assessment.needs_intervention or assessment.health_status == "unhealthy":
            return max(30, base_sleep // 2)

        return base_sleep

    def _log_assessment(self, assessment: Dict[str, Any]):
        """Log self-assessment"""
        self.logger.info(f"  Health: {assessment.health_status}")
        self.logger.info(f"  Coverage: {assessment.test_coverage:.1%}")
        self.logger.info(f"  Complexity: {assessment.avg_complexity:.1f}")
        self.logger.info(f"  Success Rate: {assessment.improvement_success_rate:.1%}")
        self.logger.info(f"  Active Goals: {assessment.goals_active}")

    def _log_new_goals(self, goals: List):
        """Log new goals"""
        self.logger.info(f"  Generated {len(goals)} new goals:")
        for goal in goals[:3]:  # Top 3
            self.logger.info(f"    - {goal.description} (priority: {goal.priority:.2f})")

    def _log_cycle_plan(self, cycle_plan: Dict[str, Any]):
        """Log cycle plan"""
        self.logger.info(f"  Tasks: {len(cycle_plan.get('tasks', []))}")
        self.logger.info(f"  Files: {len(cycle_plan.get('files_to_improve', []))}")
        self.logger.info(f"  Learning: {len(cycle_plan.get('learning_topics', []))}")

    def _log_outcomes(self, outcomes: List[Dict[str, Any]]):
        """Log cycle outcomes"""
        successes = sum(1 for o in outcomes if o.get('success'))
        self.logger.info(f"  Outcomes: {successes}/{len(outcomes)} successful")

    def _log_shutdown(self):
        """Log shutdown statistics"""
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"AUTONOMOUS CONSCIOUSNESS SHUTDOWN")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"  Total Cycles: {self.current_cycle}")
            self.logger.info(f"  Runtime: {runtime:.0f}s ({runtime/3600:.1f}h)")
            self.logger.info(f"  Avg Cycle Time: {runtime/self.current_cycle:.1f}s")
            self.logger.info(f"{'='*60}")

    def _load_config(self, config_path: str):
        """Load configuration from file"""
        import json
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config.get('autonomous_mode', {}))
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}, using defaults")

    def pause(self):
        """Pause the consciousness loop"""
        self.is_paused = True
        self.logger.warning("Consciousness loop PAUSED")

    def resume(self):
        """Resume the consciousness loop"""
        self.is_paused = False
        self.logger.info("Consciousness loop RESUMED")

    def stop(self):
        """Stop the consciousness loop"""
        self.is_running = False
        self.logger.warning("Consciousness loop STOPPING...")

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_cycle': self.current_cycle,
            'runtime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'active_agents': len(self.agents),
            'active_goals': len(self.goal_engine.active_goals),
            'latest_assessment': self.self_monitor.get_latest_assessment()
        }


async def main():
    """Main entry point for autonomous consciousness"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create consciousness
    consciousness = AutonomousConsciousness()

    try:
        # Run the autonomous loop
        await consciousness.consciousness_loop()
    except KeyboardInterrupt:
        consciousness.logger.info("\nShutdown requested")
        consciousness.stop()


if __name__ == "__main__":
    # Run the autonomous consciousness
    asyncio.run(main())
