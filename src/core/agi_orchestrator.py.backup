#!/usr/bin/env python3
"""
HUMAN 2.0 AGI Orchestrator
Master system that coordinates:
- Self-improvement (code modification)
- Autonomous learning (web knowledge)
- Self-awareness (consciousness)
- Goal planning (what to learn/improve next)

This is the brain that makes HUMAN 2.0 a true self-improving AGI.
"""

import os
import sys
import json
import logging
import asyncio
import schedule
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# V2 Systems (enhanced with full context)
from core.self_improvement_v2 import SelfImprovementV2
from core.autonomous_learner_v2 import AutonomousLearnerV2
from core.meta_learner import MetaLearner

# Consciousness systems
from consciousness.self_awareness import SelfAwarenessSystem
from consciousness.curiosity import CuriosityEngine
from consciousness.reflection import ReflectionEngine


class AGIOrchestrator:
    """
    Master AGI orchestrator

    The self-aware, self-improving, autonomous learning AI.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing HUMAN 2.0 AGI Orchestrator")

        self.config = self._load_config(config_path)

        self.self_improver = SelfImprovementV2()
        self.learner = AutonomousLearnerV2()
        self.meta_learner = MetaLearner()

        self.self_awareness = SelfAwarenessSystem()
        self.curiosity = CuriosityEngine()
        self.reflection = ReflectionEngine()

        self.is_running = False
        self.start_time = None
        self.total_cycles = 0

        self.current_goals = []
        self.completed_goals = []

        self.stats = self._initialize_stats()

        self.logger.info("AGI Orchestrator initialized successfully")

    def _initialize_stats(self) -> Dict[str, Any]:
        """Initialize statistics dictionary"""
        return {
            'total_improvements': 0,
            'total_knowledge_learned': 0,
            'total_implementations': 0,
            'total_bugs_fixed': 0,
            'uptime_hours': 0,
            'self_awareness_level': 0.0
        }

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'improvement_interval_hours': 24,
            'learning_interval_hours': 6,
            'reflection_interval_hours': 12,
            'max_files_per_improvement': 3,
            'max_questions_per_learning': 3,
            'auto_start': False
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Could not load config: {e}")

        return default_config

    def start(self):
        """Start the AGI system"""
        self._log_startup_banner()
        
        self.is_running = True
        self.start_time = datetime.now()

        self._initialize_consciousness()

        if self.config.get('auto_start'):
            self._setup_autonomous_schedules()

        self.logger.info("HUMAN 2.0 AGI is now running")
        self.logger.info(f"Started at: {self.start_time}")

    def _log_startup_banner(self):
        """Log startup banner"""
        self.logger.info("=" * 70)
        self.logger.info("STARTING HUMAN 2.0 AGI")
        self.logger.info("=" * 70)

    def stop(self):
        """Stop the AGI system"""
        self.logger.info("Stopping HUMAN 2.0 AGI...")
        self.is_running = False

        self._save_state()

        self.logger.info("HUMAN 2.0 AGI stopped")

    async def run_improvement_cycle_async(self):
        """Run one improvement cycle asynchronously"""
        self._log_cycle_header("IMPROVEMENT CYCLE", self.total_cycles + 1)

        try:
            self._update_self_awareness()

            self.logger.info("\nPhase 1: Self-Improvement (V2 with full context)")
            improvement_report = await asyncio.to_thread(
                self.self_improver.improve_codebase,
                max_files=self.config['max_files_per_improvement']
            )

            await self._process_improvement_report(improvement_report)
            self.total_cycles += 1

            return improvement_report

        except Exception as e:
            self.logger.error(f"Error in improvement cycle: {e}")
            return {'error': str(e)}

    async def _process_improvement_report(self, improvement_report: Dict[str, Any]):
        """Process improvement report and update systems"""
        self._update_improvement_stats(improvement_report)
        self._record_improvements_in_meta_learner(improvement_report)
        self._reflect_on_experience({
            'type': 'self_improvement',
            'content': improvement_report
        })

    def run_improvement_cycle(self):
        """Run one improvement cycle (sync wrapper)"""
        return asyncio.run(self.run_improvement_cycle_async())

    def _update_improvement_stats(self, improvement_report: Dict[str, Any]):
        """Update statistics from improvement report"""
        stats = improvement_report.get('stats', {})
        self.stats['total_improvements'] += stats.get('successful_improvements', 0)
        self.stats['total_bugs_fixed'] += stats.get('bugs_fixed', 0)

    def _record_improvements_in_meta_learner(self, improvement_report: Dict[str, Any]):
        """Record improvements in meta-learner for strategy optimization"""
        improvements = improvement_report.get('improvements', [])
        for imp in improvements:
            if imp.get('result'):
                self.meta_learner.record_improvement_outcome(imp['result'])

    async def run_learning_cycle_async(self):
        """Run one learning cycle asynchronously"""
        self._log_cycle_header("LEARNING CYCLE", self.learner.learning_cycle + 1)

        try:
            self._update_self_awareness()

            self.logger.info("\nPhase 2: Autonomous Learning (V2 with GitHub)")
            learning_report = await asyncio.to_thread(
                self.learner.autonomous_learning_cycle,
                max_questions=self.config['max_questions_per_learning']
            )

            await self._process_learning_report(learning_report)

            return learning_report

        except Exception as e:
            self.logger.error(f"Error in learning cycle: {e}")
            return {'error': str(e)}

    async def _process_learning_report(self, learning_report: Dict[str, Any]):
        """Process learning report and update systems"""
        self._update_learning_stats(learning_report)
        self._reflect_on_experience({
            'type': 'autonomous_learning',
            'content': learning_report
        })

    def run_learning_cycle(self):
        """Run one learning cycle (sync wrapper)"""
        return asyncio.run(self.run_learning_cycle_async())

    def _update_learning_stats(self, learning_report: Dict[str, Any]):
        """Update statistics from learning report"""
        self.stats['total_knowledge_learned'] += learning_report.get('learned_this_cycle', 0)
        self.stats['total_implementations'] += learning_report.get('total_implemented', 0)

    async def run_full_cycle_async(self):
        """Run complete cycle: learn, improve, reflect (async)"""
        self._log_cycle_header("FULL AGI CYCLE", self.total_cycles + 1, separator="#")

        results = self._initialize_cycle_results()

        results['learning'] = await self.run_learning_cycle_async()
        results['improvement'] = await self.run_improvement_cycle_async()
        results['reflection'] = await asyncio.to_thread(self._run_reflection_cycle)

        if self._should_run_meta_learning():
            self.logger.info("\nPhase 4: Meta-Learning Optimization")
            results['meta_learning'] = await asyncio.to_thread(self._run_meta_learning_optimization)

        self._finalize_cycle(results)

        return results

    def _finalize_cycle(self, results: Dict[str, Any]):
        """Finalize cycle by updating stats and generating reports"""
        self._update_stats()
        self._save_state()
        self._generate_comprehensive_report(results)

    def run_full_cycle(self):
        """Run complete cycle: learn, improve, reflect (sync wrapper)"""
        return asyncio.run(self.run_full_cycle_async())

    def _initialize_cycle_results(self) -> Dict[str, Any]:
        """Initialize results dictionary for cycle"""
        return {
            'cycle': self.total_cycles + 1,
            'timestamp': datetime.now().isoformat(),
            'learning': None,
            'improvement': None,
            'reflection': None,
            'goals_completed': []
        }

    def _should_run_meta_learning(self) -> bool:
        """Determine if meta-learning should run"""
        return self.total_cycles > 0 and self.total_cycles % 5 == 0

    def _log_cycle_header(self, title: str, cycle_num: int, separator: str = "="):
        """Log cycle header"""
        self.logger.info("\n" + separator * 70)
        self.logger.info(f"{title} {cycle_num}")
        self.logger.info(separator * 70)

    def _initialize_consciousness(self):
        """Initialize consciousness systems"""
        self.logger.info("Initializing consciousness...")

        self.self_awareness.initialize()

        self.self_awareness.add_experience({
            'type': 'system_start',
            'content': 'HUMAN 2.0 AGI system started',
            'timestamp': datetime.now().isoformat(),
            'capabilities': self._get_system_capabilities()
        })

        self.logger.info("Consciousness initialized")

    def _get_system_capabilities(self) -> List[str]:
        """Get list of system capabilities"""
        return [
            'self_improvement',
            'autonomous_learning',
            'self_awareness',
            'curiosity_driven_exploration',
            'code_modification',
            'web_knowledge_acquisition'
        ]

    def _update_self_awareness(self):
        """Update self-awareness state"""
        self.self_awareness.update_state()
        self.self_awareness.current_goals = self.current_goals
        self.stats['self_awareness_level'] = self._calculate_self_awareness_level()

    def _calculate_self_awareness_level(self) -> float:
        """Calculate current level of self-awareness"""
        level_components = [
            (len(self.self_awareness.personal_history) > 0, 0.2),
            (len(self.reflection.known_patterns) > 0, 0.2),
            (len(self.current_goals) > 0, 0.2),
            (self.stats['total_improvements'] > 0, 0.2),
            (self.stats['total_knowledge_learned'] > 0, 0.2)
        ]

        level = sum(weight for condition, weight in level_components if condition)
        return min(1.0, level)

    def _reflect_on_experience(self, experience: Dict[str, Any]):
        """Reflect on an experience"""
        self.reflection.process_experience(experience)
        self.self_awareness.add_experience(experience)

    def _run_reflection_cycle(self) -> Dict[str, Any]:
        """Run reflection cycle"""
        self.logger.info("\nPhase 3: Reflection")
        return self.self_awareness.reflect()

    def _setup_autonomous_schedules(self):
        """Set up autonomous operation schedules"""
        self.logger.info("Setting up autonomous schedules...")

        schedule.every(self.config['improvement_interval_hours']).hours.do(
            self.run_improvement_cycle
        )

        schedule.every(self.config['learning_interval_hours']).hours.do(
            self.run_learning_cycle
        )

        schedule.every(self.config['reflection_interval_hours']).hours.do(
            self._run_reflection_cycle
        )

        self.logger.info("Autonomous schedules configured")

    def _update_stats(self):
        """Update statistics"""
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            self.stats['uptime_hours'] = round(uptime, 2)

    def _save_state(self):
        """Save current state"""
        state_file = Path('state') / 'agi_state.json'
        state_file.parent.mkdir(exist_ok=True)

        state = {
            'timestamp': datetime.now().isoformat(),
            'total_cycles': self.total_cycles,
            'stats': self.stats,
            'current_goals': self.current_goals,
            'completed_goals': self.completed_goals,
            'is_running': self.is_running
        }

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive AGI report"""
        report_file = Path('reports') / f'agi_cycle_{self.total_cycles}.json'
        report_file.parent.mkdir(exist_ok=True)

        report = self._build_report(results)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"\nComprehensive report saved to: {report_file}")

    def _build_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive report dictionary"""
        return {
            'cycle': results['cycle'],
            'timestamp': results['timestamp'],
            'results': results,
            'stats': self.stats,
            'self_awareness_level': self.stats['self_awareness_level'],
            'system_status': 'operational' if self.is_running else 'stopped'
        }

    def _run_meta_learning_optimization(self) -> Dict[str, Any]:
        """Run meta-learning optimization"""
        try:
            meta_stats = self.meta_learner.get_statistics()
            self.logger.info(f"Meta-learner stats: {meta_stats}")

            total_improvements = meta_stats.get('total_improvements', 0)
            
            if total_improvements >= 10:
                optimized_params = self.meta_learner.optimize_strategies()
                self.logger.info(f"Meta-learning optimization complete: {optimized_params}")
                return optimized_params
            
            self.logger.info(f"Insufficient data for optimization ({total_improvements} < 10)")
            return {'status': 'insufficient_data', 'stats': meta_stats}

        except Exception as e:
            self.logger.error(f"Error in meta-learning optimization: {e}")
            return {'error': str(e)}