#!/usr/bin/env python3
"""
HUMAN 2.0 - Tester Agent
Auto-generates tests and validates code improvements.

Responsibilities:
- Auto-generate unit tests for untested code
- Run existing tests
- Validate improvements don't break tests
- Track test coverage
- Report test results
"""

import ast
import os
import subprocess
import logging
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
class TestResult:
    """Result of test execution"""
    file_path: str
    tests_run: int
    tests_passed: int
    tests_failed: int
    coverage: float  # 0-1
    execution_time: float  # seconds
    errors: List[str]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def pass_rate(self) -> float:
        if self.tests_run == 0:
            return 0.0
        return self.tests_passed / self.tests_run


class TesterAgent(BaseAgent):
    """
    Agent responsible for testing and test generation.

    Subscribes to:
    - improvement_proposed: Validate before applying
    - code_analyzed: Generate tests if missing
    - improvement_applied: Run tests after changes

    Publishes:
    - tests_generated: New tests created
    - tests_executed: Test results available
    - test_coverage_updated: Coverage metrics updated
    """

    def __init__(self, name: str, event_bus: EventBus, resources: SharedResources):
        """
        Initialize Tester Agent.

        Args:
            name: Agent name
            event_bus: Event bus for communication
            resources: Shared resources
        """
        super().__init__(name, event_bus)
        self.resources = resources

        # Configuration
        self.config = {
            'test_dir': 'tests',
            'min_coverage': 0.75,
            'auto_generate_tests': True,
            'pytest_args': ['-v', '--tb=short']
        }

        self.logger.info(f"TesterAgent initialized with config: {self.config}")

    def register_event_handlers(self):
        """Register event handlers"""
        self.event_bus.subscribe(EventTypes.IMPROVEMENT_PROPOSED, self.on_improvement_proposed, self.name)
        self.event_bus.subscribe(EventTypes.CODE_ANALYZED, self.on_code_analyzed, self.name)
        self.event_bus.subscribe(EventTypes.IMPROVEMENT_APPLIED, self.on_improvement_applied, self.name)
        self.logger.info(f"Subscribed to: improvement_proposed, code_analyzed, improvement_applied")

    async def on_improvement_proposed(self, event: Event):
        """Validate improvement doesn't break tests"""
        improvement = event.data.get('improvement')
        file_path = improvement.get('file_path')

        self.logger.info(f"Validating improvement for: {file_path}")

        # Run tests for this file
        task = {
            'type': 'validate',
            'file_path': file_path,
            'improvement': improvement
        }
        result = await self.execute_task(task)

    async def on_code_analyzed(self, event: Event):
        """Generate tests if missing"""
        analysis = event.data.get('analysis')
        file_path = event.data.get('file_path')

        # Check if file has tests
        if self._has_tests(file_path):
            self.logger.debug(f"{file_path} already has tests")
            return

        if self.config['auto_generate_tests']:
            self.logger.info(f"Generating tests for: {file_path}")
            task = {
                'type': 'generate',
                'file_path': file_path,
                'analysis': analysis
            }
            result = await self.execute_task(task)

    async def on_improvement_applied(self, event: Event):
        """Run tests after improvement"""
        file_path = event.data.get('file_path')

        self.logger.info(f"Running tests after improvement: {file_path}")

        task = {
            'type': 'run',
            'file_path': file_path
        }
        result = await self.execute_task(task)

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing: Generate or run tests.

        Args:
            task: Task with type (generate/run/validate) and file_path

        Returns:
            Test result
        """
        task_type = task.get('type')
        file_path = task['file_path']

        if task_type == 'generate':
            return await self._generate_tests(file_path, task.get('analysis'))
        elif task_type == 'run':
            return await self._run_tests(file_path)
        elif task_type == 'validate':
            return await self._validate_improvement(file_path, task.get('improvement'))
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _generate_tests(self, file_path: str, analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate tests for a file using existing test generator.

        Args:
            file_path: Path to file to test
            analysis: Optional analysis results

        Returns:
            Generation result
        """
        self.logger.info(f"Generating tests for: {file_path}")

        try:
            # Use the existing TestGenerator from V2
            from core.test_generator import TestGenerator

            test_gen = TestGenerator()
            result = test_gen.generate_tests_for_file(file_path)

            if result.get('success'):
                # Publish tests generated event
                await self.publish_event(
                    EventTypes.TESTS_GENERATED,
                    {
                        'file_path': file_path,
                        'test_file': result.get('test_file'),
                        'num_tests': result.get('num_tests', 0)
                    },
                    EventPriority.NORMAL
                )

                return {
                    'success': True,
                    'test_file': result.get('test_file'),
                    'num_tests': result.get('num_tests', 0)
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }

        except Exception as e:
            self.logger.error(f"Failed to generate tests: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _run_tests(self, file_path: str) -> Dict[str, Any]:
        """
        Run tests for a file.

        Args:
            file_path: Path to file

        Returns:
            Test results
        """
        test_file = self._get_test_file(file_path)

        if not test_file or not test_file.exists():
            self.logger.warning(f"No test file found for: {file_path}")
            return {
                'success': False,
                'error': 'No test file found'
            }

        self.logger.info(f"Running tests: {test_file}")

        try:
            # Run pytest
            cmd = ['pytest', str(test_file)] + self.config['pytest_args']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse output
            test_result = self._parse_pytest_output(result.stdout, result.stderr)
            test_result.file_path = file_path

            # Publish test results
            await self.publish_event(
                EventTypes.TESTS_EXECUTED,
                {
                    'file_path': file_path,
                    'test_file': str(test_file),
                    'result': asdict(test_result)
                },
                EventPriority.HIGH if test_result.tests_failed > 0 else EventPriority.NORMAL
            )

            # Update coverage
            await self.publish_event(
                EventTypes.TEST_COVERAGE_UPDATED,
                {
                    'file_path': file_path,
                    'coverage': test_result.coverage
                },
                EventPriority.LOW
            )

            return {
                'success': True,
                'result': asdict(test_result)
            }

        except subprocess.TimeoutExpired:
            self.logger.error(f"Tests timed out: {test_file}")
            return {
                'success': False,
                'error': 'Tests timed out'
            }
        except Exception as e:
            self.logger.error(f"Failed to run tests: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _validate_improvement(self, file_path: str, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate improvement by running tests.

        Args:
            file_path: Path to file
            improvement: Improvement to validate

        Returns:
            Validation result
        """
        self.logger.info(f"Validating improvement: {file_path}")

        # Temporarily apply improvement
        original_code = improvement.get('original_code')
        improved_code = improvement.get('improved_code')

        try:
            # Backup
            with open(file_path, 'r', encoding='utf-8') as f:
                current_code = f.read()

            # Apply improvement temporarily
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(improved_code)

            # Run tests
            test_result = await self._run_tests(file_path)

            # Restore original
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(current_code)

            # Check if tests passed
            if test_result.get('success'):
                result_data = test_result.get('result', {})
                if result_data.get('tests_failed', 0) == 0:
                    return {
                        'success': True,
                        'valid': True,
                        'test_result': result_data
                    }
                else:
                    return {
                        'success': True,
                        'valid': False,
                        'reason': f"{result_data.get('tests_failed')} tests failed",
                        'test_result': result_data
                    }
            else:
                return {
                    'success': True,
                    'valid': False,
                    'reason': test_result.get('error', 'Tests failed to run')
                }

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            # Restore original on error
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(current_code)
            except:
                pass
            return {
                'success': False,
                'error': str(e)
            }

    def _has_tests(self, file_path: str) -> bool:
        """Check if file has tests"""
        test_file = self._get_test_file(file_path)
        return test_file is not None and test_file.exists()

    def _get_test_file(self, file_path: str) -> Optional[Path]:
        """Get test file path for a source file"""
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name

        # Try test_<filename>.py in tests/ directory
        test_file = Path(self.config['test_dir']) / f'test_{file_name}'

        if test_file.exists():
            return test_file

        # Try in same directory
        test_file = file_path_obj.parent / f'test_{file_name}'
        if test_file.exists():
            return test_file

        return None

    def _parse_pytest_output(self, stdout: str, stderr: str) -> TestResult:
        """Parse pytest output to extract metrics"""
        result = TestResult(
            file_path='',
            tests_run=0,
            tests_passed=0,
            tests_failed=0,
            coverage=0.0,
            execution_time=0.0,
            errors=[]
        )

        # Parse test counts from output
        # Example: "5 passed in 0.05s" or "2 failed, 3 passed in 0.10s"
        import re

        # Find test counts
        passed_match = re.search(r'(\d+) passed', stdout)
        failed_match = re.search(r'(\d+) failed', stdout)
        time_match = re.search(r'in ([\d.]+)s', stdout)

        if passed_match:
            result.tests_passed = int(passed_match.group(1))
        if failed_match:
            result.tests_failed = int(failed_match.group(1))
        if time_match:
            result.execution_time = float(time_match.group(1))

        result.tests_run = result.tests_passed + result.tests_failed

        # Extract errors
        if 'FAILED' in stdout:
            for line in stdout.split('\n'):
                if 'FAILED' in line or 'ERROR' in line:
                    result.errors.append(line.strip())

        # Simple coverage estimate (would need pytest-cov for real coverage)
        if result.tests_run > 0:
            result.coverage = result.tests_passed / result.tests_run

        return result

    def validate_output(self, output: Any) -> bool:
        """Validate test output"""
        if not isinstance(output, dict):
            return False

        if not output.get('success') and 'error' not in output:
            return False

        return True


if __name__ == "__main__":
    # Test tester agent
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus, create_event
    from core.shared_resources import SharedResources

    async def test_tester():
        """Test tester agent"""
        bus = EventBus()
        resources = SharedResources()
        agent = TesterAgent("tester", bus, resources)

        # Test running existing tests
        task = {
            'type': 'run',
            'file_path': 'src/agents/base_agent.py'
        }

        result = await agent.execute_task(task)
        print(f"\nTest Run Result:")
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            test_result = result.get('result', {})
            print(f"  Tests Run: {test_result.get('tests_run')}")
            print(f"  Tests Passed: {test_result.get('tests_passed')}")
            print(f"  Tests Failed: {test_result.get('tests_failed')}")
            print(f"  Pass Rate: {test_result.get('pass_rate', 0):.1%}")

        # Test event subscription
        await bus.publish(create_event(
            EventTypes.CODE_ANALYZED,
            {
                'file_path': 'src/agents/base_agent.py',
                'analysis': {'priority_score': 0.5}
            },
            'analyzer'
        ))

        await asyncio.sleep(0.5)

        # Get status
        status = agent.get_status()
        print(f"\nAgent Status:")
        print(f"  Name: {status['name']}")
        print(f"  Tasks Processed: {status['metrics']['tasks_processed']}")

    asyncio.run(test_tester())
